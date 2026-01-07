"""
シミュレーション実行（所有者優先＋ゲストプリエンプト）
所有者は自分のGPUでゲスト実行中でも割り込み可能（プリエンプト）
プリエンプトされたゲストは、以下から動的に選択して再開する：
 1) 自分のGPUに移動（ゲストは無視できる、所有者タスクのみ待つ）
 2) プリエンプト元GPUで所有者完了まで先頭待機
 3) 他のGPUのキュー末尾に並ぶ（オーナー到来による中断リスクを期待値で加味）
プリエンプトされたタスクは残作業量から再開する。
"""

import numpy as np
import heapq
from definitions import User, GPU, Task
from config import (
    NUM_USERS,
    ARRIVAL_RATE,
    ARRIVAL_RATES,
    SIMULATION_TIME,
    RANDOM_SEED,
    GPU_PERFORMANCE_LEVELS,
    GPU_TIER_ASSIGNMENT,
    TASK_SIZE_MEAN,
    TASK_SIZE_MEANS,
    TASK_SIZE_MEAN_GLOBAL,
)
from results import analyze_and_print_results
from task_patterns import load_patterns, save_patterns
import os

np.random.seed(RANDOM_SEED)


class SimulatorWithOwnerPreemption:
    """
    所有者優先＋ゲストプリエンプトの共有GPU版シミュレータ
    """
    def __init__(self, task_patterns=None):
        self.users = []
        self.shared_gpus = []
        self.gpu_owner = {}
        self.event_queue = []  # (time, event_type, data)
        self.current_time = 0.0
        self.all_tasks = []
        self.task_patterns = task_patterns or {}

    # ---------------------- 基本セットアップ ----------------------
    def initialize(self):
        # 共有GPUプール作成
        for user_id in range(NUM_USERS):
            tier = None
            for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
                if user_id in user_list:
                    tier = tier_name
                    break
            rate = GPU_PERFORMANCE_LEVELS[tier]
            gpu = GPU(gpu_id=user_id, processing_rate=rate)
            self.shared_gpus.append(gpu)
            self.gpu_owner[user_id] = user_id

        # ユーザー作成（共有プール運用）
        for user_id in range(NUM_USERS):
            arrival_rate = ARRIVAL_RATES.get(str(user_id), ARRIVAL_RATE)
            user = User(user_id=user_id, gpu=None, arrival_rate=arrival_rate)
            self.users.append(user)

            arrivals = self.task_patterns.get("arrivals", {}).get(str(user_id), [])
            if arrivals:
                self.schedule_event(arrivals[0], "task_arrival", user_id)
            else:
                first_arrival = np.random.exponential(1.0 / arrival_rate)
                self.schedule_event(first_arrival, "task_arrival", user_id)

    def schedule_event(self, time, event_type, data):
        heapq.heappush(self.event_queue, (time, event_type, data))

    # ---------------------- 実効性能・待ち時間推定 ----------------------
    def get_owner_utilization(self, gpu):
        owner_id = self.gpu_owner[gpu.gpu_id]
        owner_lambda = ARRIVAL_RATES.get(str(owner_id), ARRIVAL_RATE)
        owner_task_size_mean = TASK_SIZE_MEANS.get(str(owner_id), TASK_SIZE_MEAN)
        return owner_lambda * owner_task_size_mean / gpu.processing_rate

    def get_effective_processing_rate(self, gpu, user_id):
        if self.gpu_owner[gpu.gpu_id] == user_id:
            return gpu.processing_rate
        else:
            rho_own = self.get_owner_utilization(gpu)
            return gpu.processing_rate / (1.0 + rho_own)

    def compute_job_size(self, user_id, arrival_time):
        job_sizes = self.task_patterns.get("job_sizes", {}).get(str(user_id), {})
        job_size = job_sizes.get(str(arrival_time))
        if job_size is None:
            mean = TASK_SIZE_MEANS.get(str(user_id), TASK_SIZE_MEAN)
            job_size = np.random.exponential(mean)
        return job_size

    def predict_owner_wait_on_gpu(self, gpu, owner_id):
        # 実行中が所有者ならその残りを待つ、ゲストなら待たない
        wait = 0.0
        if gpu.current_task is not None:
            if gpu.current_task.user_id == owner_id:
                wait += max(0.0, gpu.finish_time - self.current_time)
        for t in gpu.task_queue:
            if t.user_id == owner_id:
                # 所有者キュー分
                size = self.compute_job_size(t.user_id, t.arrival_time)
                wait += size / gpu.processing_rate
        return wait

    def expected_interruption_penalty(self, gpu, service_time):
        # 所有者到来率による途中切断リスクの期待ペナルティ
        owner_id = self.gpu_owner[gpu.gpu_id]
        lam = ARRIVAL_RATES.get(str(owner_id), ARRIVAL_RATE)
        mean_owner_size = TASK_SIZE_MEANS.get(str(owner_id), TASK_SIZE_MEAN)
        mean_owner_service = mean_owner_size / gpu.processing_rate
        p_interrupt = 1.0 - np.exp(-lam * service_time)
        penalty = p_interrupt * mean_owner_service
        return penalty

    # ---------------------- GPU選択ロジック ----------------------
    def predict_completion_time_own_gpu(self, gpu, user_id, remaining_work):
        wait_owner = self.predict_owner_wait_on_gpu(gpu, user_id)
        return self.current_time + wait_owner + remaining_work / gpu.processing_rate

    def predict_completion_time_other_gpu(self, gpu, user_id, remaining_work):
        base = gpu.finish_time if gpu.current_task is not None else self.current_time
        # 既存キューの概算（全タスクを平均サイズで代表）
        qlen = len(gpu.task_queue)
        mu_eff = self.get_effective_processing_rate(gpu, user_id)
        queue_time = (TASK_SIZE_MEAN_GLOBAL / mu_eff) * qlen if qlen > 0 else 0.0
        service_time = remaining_work / mu_eff
        penalty = self.expected_interruption_penalty(gpu, service_time)
        return base + queue_time + service_time + penalty

    def select_best_gpu_for_new(self, user_id, remaining_work):
        best_gpu = None
        best_time = float('inf')
        for gpu in self.shared_gpus:
            if self.gpu_owner[gpu.gpu_id] == user_id:
                t = self.predict_completion_time_own_gpu(gpu, user_id, remaining_work)
            else:
                t = self.predict_completion_time_other_gpu(gpu, user_id, remaining_work)
            if t < best_time:
                best_time = t
                best_gpu = gpu
        return best_gpu

    def select_after_preempt(self, task, preempt_gpu):
        # 1) 自分のGPU
        own_gpu = self.shared_gpus[task.user_id]
        t_own = self.predict_completion_time_own_gpu(own_gpu, task.user_id, task.remaining_work)

        # 2) プリエンプト元GPUで先頭待機（所有者待ち）
        wait_owner = self.predict_owner_wait_on_gpu(preempt_gpu, self.gpu_owner[preempt_gpu.gpu_id])
        t_wait_here = self.current_time + wait_owner + task.remaining_work / preempt_gpu.processing_rate

        # 3) 他GPUのキュー末尾（期待ペナルティ込み）から最良を探す
        best_other_time = float('inf')
        best_other_gpu = None
        for gpu in self.shared_gpus:
            if gpu is preempt_gpu:
                continue
            t = self.predict_completion_time_other_gpu(gpu, task.user_id, task.remaining_work)
            if t < best_other_time:
                best_other_time = t
                best_other_gpu = gpu

        # 比較して最短の行き先を返す
        choices = [(t_own, 'own', own_gpu), (t_wait_here, 'wait_here', preempt_gpu), (best_other_time, 'other', best_other_gpu)]
        return min(choices, key=lambda x: x[0])

    # ---------------------- 実行・プリエンプト ----------------------
    def start_task_on_gpu(self, gpu, task):
        task.start_time = self.current_time
        gpu.current_task = task
        # 所有者以外は実効レートを適用して遅延を反映
        if self.gpu_owner[gpu.gpu_id] == task.user_id:
            rate = gpu.processing_rate
        else:
            rate = self.get_effective_processing_rate(gpu, task.user_id)
        service_time = task.remaining_work / rate
        finish_time = self.current_time + service_time
        gpu.finish_time = finish_time
        self.schedule_event(finish_time, "gpu_finish", gpu.gpu_id)

    def preempt_guest_if_needed(self, gpu, owner_id):
        if gpu.current_task is not None and gpu.current_task.user_id != owner_id:
            # ゲストをプリエンプト
            guest = gpu.current_task
            elapsed = max(0.0, self.current_time - (guest.start_time or self.current_time))
            processed_work = elapsed * gpu.processing_rate
            # 初回実行時に残作業が未設定なら設定
            if getattr(guest, 'remaining_work', None) is None:
                # 到着時刻でサイズ取得（初回開始時に設定される想定）
                guest.remaining_work = self.compute_job_size(guest.user_id, guest.arrival_time)
            guest.remaining_work = max(0.0, guest.remaining_work - processed_work)

            # プリエンプト状態へ：GPUから降ろす
            gpu.current_task = None

            # ゲストの次の行き先を決める
            best_time, choice, dest_gpu = self.select_after_preempt(guest, gpu)
            if choice == 'own':
                dest_gpu.add_task(guest, owner_id=guest.user_id)
                if dest_gpu.current_task is None:
                    self.start_task_on_gpu(dest_gpu, guest)
            elif choice == 'wait_here':
                gpu.task_queue.insert(0, guest)  # 先頭で待機
            else:  # other
                dest_gpu.add_task(guest)  # 末尾
                if dest_gpu.current_task is None:
                    self.start_task_on_gpu(dest_gpu, guest)

    # ---------------------- イベント処理 ----------------------
    def process_task_arrival(self, user_id):
        user = self.users[user_id]
        task = user.create_task(self.current_time)
        # タスクサイズ→残作業として保持
        task.remaining_work = self.compute_job_size(user_id, task.arrival_time)
        self.all_tasks.append(task)

        # 最適GPU選択（他GPUは中断リスク期待値込み）
        best_gpu = self.select_best_gpu_for_new(user_id, task.remaining_work)
        task.assigned_gpu = best_gpu

        # 自分のGPUを選ぶ場合、ゲストが走っていればプリエンプト
        if self.gpu_owner[best_gpu.gpu_id] == user_id:
            self.preempt_guest_if_needed(best_gpu, owner_id=user_id)

        # キューへ投入（所有者優先）
        best_gpu.add_task(task, owner_id=self.gpu_owner[best_gpu.gpu_id])

        # 空いていれば開始
        if best_gpu.current_task is None:
            self.start_task_on_gpu(best_gpu, task)

        # 次の到着イベント
        arrivals = self.task_patterns.get("arrivals", {}).get(str(user_id), [])
        next_idx = len([t for t in user.tasks if t is not None])
        if next_idx < len(arrivals):
            next_t = arrivals[next_idx]
            if next_t <= SIMULATION_TIME:
                self.schedule_event(next_t, "task_arrival", user_id)

    def process_gpu_finish(self, gpu_id):
        gpu = None
        for g in self.shared_gpus:
            if g.gpu_id == gpu_id:
                gpu = g
                break
        if gpu is None:
            return

        task = gpu.current_task
        if task is None:
            return
        task.completion_time = self.current_time
        task.remaining_work = 0.0
        gpu.current_task = None

        # 次があれば開始
        if len(gpu.task_queue) > 0:
            next_task = gpu.task_queue.pop(0)
            self.start_task_on_gpu(gpu, next_task)

    # ---------------------- 実行ループ ----------------------
    def run(self):
        self.initialize()
        while self.event_queue and self.current_time <= SIMULATION_TIME:
            time, event_type, data = heapq.heappop(self.event_queue)
            self.current_time = time
            if event_type == "task_arrival":
                self.process_task_arrival(data)
            elif event_type == "gpu_finish":
                self.process_gpu_finish(data)

        print(f"シミュレーション終了：時刻 {self.current_time}")
        print(f"発生したタスク総数：{len(self.all_tasks)}")
        return self.all_tasks


def main():
    # タスクパターン生成／読み込み
    if not os.path.exists("task_patterns.json"):
        print("タスクパターンを生成中...")
        save_patterns()
    patterns = load_patterns()

    # 実行
    sim = SimulatorWithOwnerPreemption(task_patterns=patterns)
    tasks = sim.run()

    # 結果
    analyze_and_print_results(tasks, NUM_USERS, mode="with_sharing_owner_preemption")


if __name__ == "__main__":
    main()
