"""
シミュレーション実行（所有者優先＋ゲストプリエンプト）
18ユーザー・9ティア構成で実行
所有者は自分のGPUでゲスト実行中でも割り込み可能（プリエンプト）
プリエンプトされたゲストは、以下から動的に選択して再開する：
 1) 自分のGPUに移動
 2) プリエンプト元GPUで所有者完了まで先頭待機
 3) 他のGPUのキュー末尾に並ぶ
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
    TASK_SIZE_MEANS,
    TASK_SIZE_MEAN_GLOBAL,
    BATCH_SIZES,
    EPOCHS,
    INTERRUPTION_OVERHEAD_FACTOR,
)
from results import analyze_and_print_results
from task_patterns import load_patterns, save_patterns
import os

np.random.seed(RANDOM_SEED)


class SimulatorWithOwnerPreemption:
    """
    所有者優先＋ゲストプリエンプトの共有GPU版シミュレータ
    """
    def __init__(self, task_patterns=None, participating_users=None):
        self.users = []
        self.shared_gpus = []
        self.gpu_owner = {}
        self.event_queue = []  # (time, event_type, data)
        self.current_time = 0.0
        self.all_tasks = []
        self.task_patterns = task_patterns or {}
        self.participating_users = participating_users if participating_users is not None else list(range(NUM_USERS))
        # デバッグ用カウンタ
        self.preemption_count = 0
        self.realloc_own = 0
        self.realloc_wait_here = 0
        self.realloc_other = 0

    # ---------------------- 基本セットアップ ----------------------
    def initialize(self):
        # 共有GPUプール作成（参加ユーザーのGPUのみ）
        for user_id in self.participating_users:
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
        batch_size = BATCH_SIZES.get(owner_id, 1000)
        epochs = EPOCHS.get(owner_id, 1)
        owner_task_size_mean = TASK_SIZE_MEANS.get(owner_id, TASK_SIZE_MEAN_GLOBAL) * batch_size * epochs
        return owner_lambda * owner_task_size_mean / gpu.processing_rate

    def get_effective_processing_rate(self, gpu, user_id):
        if self.gpu_owner[gpu.gpu_id] == user_id:
            return gpu.processing_rate
        else:
            rho_own = self.get_owner_utilization(gpu)
            return gpu.processing_rate / (1.0 + rho_own)

    def compute_job_size(self, user_id, arrival_time):
        job_sizes = self.task_patterns.get("sizes", {}).get(str(user_id), {})
        job_size = job_sizes.get(str(arrival_time))
        if job_size is None:
            base_size = TASK_SIZE_MEANS.get(user_id, TASK_SIZE_MEAN_GLOBAL)
            batch_size = BATCH_SIZES.get(user_id, 1000)
            epochs = EPOCHS.get(user_id, 1)
            mean = base_size * batch_size * epochs
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
        base_size = TASK_SIZE_MEANS.get(owner_id, TASK_SIZE_MEAN_GLOBAL)
        batch_size = BATCH_SIZES.get(owner_id, 1000)
        epochs = EPOCHS.get(owner_id, 1)
        mean_owner_size = base_size * batch_size * epochs
        mean_owner_service = mean_owner_size / gpu.processing_rate
        # 期待割込み回数（Poissonの期待値）：lam * service_time を用いる強化版
        expected_interruptions = lam * service_time
        # 追加の再開・マイグレーションオーバーヘッドを係数で加味（強めに設定）
        penalty = expected_interruptions * (mean_owner_service * (1.0 + INTERRUPTION_OVERHEAD_FACTOR))
        return penalty

    # ---------------------- GPU選択ロジック ----------------------
    def predict_completion_time_own_gpu(self, gpu, user_id, remaining_work):
        wait_owner = self.predict_owner_wait_on_gpu(gpu, user_id)
        return self.current_time + wait_owner + remaining_work / gpu.processing_rate

    def predict_completion_time_other_gpu(self, gpu, user_id, remaining_work):
        base = gpu.finish_time if gpu.current_task is not None else self.current_time
        # 既存キューの正確な計算（各タスクの実際のサイズを使用、バッチ係数適用）
        mu_eff = self.get_effective_processing_rate(gpu, user_id)
        queue_time = 0.0
        for task in gpu.task_queue:
            base_size = TASK_SIZE_MEANS.get(task.user_id, TASK_SIZE_MEAN_GLOBAL)
            batch_size = BATCH_SIZES.get(task.user_id, 1000)
            epochs = EPOCHS.get(task.user_id, 1)
            task_size_mean = base_size * batch_size * epochs
            queue_time += task_size_mean / mu_eff
        service_time = remaining_work / mu_eff
        penalty = self.expected_interruption_penalty(gpu, service_time)
        return base + queue_time + service_time + penalty

    def select_best_gpu_for_new(self, user_id, remaining_work):
        if not self.shared_gpus:
            return None
        
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
        # 1) 自分のGPU（参加していない場合はスキップ）
        own_gpu = None
        for gpu in self.shared_gpus:
            if self.gpu_owner[gpu.gpu_id] == task.user_id:
                own_gpu = gpu
                break
        
        if own_gpu is None:
            # 自分のGPUがない場合は他の選択肢のみを検討
            t_own = float('inf')
        else:
            t_own = self.predict_completion_time_own_gpu(own_gpu, task.user_id, task.remaining_work)

        # 2) プリエンプト元GPUで先頭待機（所有者待ち）
        wait_owner = self.predict_owner_wait_on_gpu(preempt_gpu, self.gpu_owner[preempt_gpu.gpu_id])
        # ゲストはμ_effで処理されるためサービス率を実効値に
        mu_eff_here = self.get_effective_processing_rate(preempt_gpu, task.user_id)
        service_time_here = task.remaining_work / mu_eff_here
        # 再開後の割り込みリスクを考慮
        penalty_here = self.expected_interruption_penalty(preempt_gpu, service_time_here)
        t_wait_here = self.current_time + wait_owner + service_time_here + penalty_here

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
        # 安全チェック：GPUが空いていることを確認
        if gpu.current_task is not None:
            raise RuntimeError(f"GPU {gpu.gpu_id} already has a running task {gpu.current_task.task_id}")
        
        # 初回実行時のみstart_timeを設定（プリエンプト後の再開では上書きしない）
        if not hasattr(task, 'start_time') or task.start_time is None:
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
        # イベントデータにGPU IDとタスクIDを含める
        self.schedule_event(finish_time, "gpu_finish", (gpu.gpu_id, task.task_id))

    def preempt_guest_if_needed(self, gpu, owner_id):
        if gpu.current_task is not None and gpu.current_task.user_id != owner_id:
            # ゲストをプリエンプト
            self.preemption_count += 1
            guest = gpu.current_task
            
            # プリエンプト回数を記録
            guest.preempted_count += 1
            
            elapsed = max(0.0, self.current_time - (guest.start_time or self.current_time))
            # 実際に用いている処理率（ゲストはμ_eff）で処理量を算出
            rate_used = self.get_effective_processing_rate(gpu, guest.user_id)
            processed_work = elapsed * rate_used
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
                self.realloc_own += 1
                dest_gpu.add_task(guest, owner_id=guest.user_id)
                guest.assigned_gpu = dest_gpu
                if dest_gpu.current_task is None:
                    self.start_task_on_gpu(dest_gpu, guest)
            elif choice == 'wait_here':
                self.realloc_wait_here += 1
                gpu.task_queue.insert(0, guest)  # 先頭で待機
                guest.assigned_gpu = gpu
            else:  # other
                self.realloc_other += 1
                dest_gpu.add_task(guest)  # 末尾
                guest.assigned_gpu = dest_gpu
                if dest_gpu.current_task is None:
                    self.start_task_on_gpu(dest_gpu, guest)

    # ---------------------- イベント処理 ----------------------
    def process_task_arrival(self, user_id):
        user = self.users[user_id]
        task = user.create_task(self.current_time)
        # タスクサイズ→残作業として保持
        task.remaining_work = self.compute_job_size(user_id, task.arrival_time)
        task.total_work = task.remaining_work
        self.all_tasks.append(task)

        # 最適GPU選択（他GPUは中断リスク期待値込み）
        best_gpu = self.select_best_gpu_for_new(user_id, task.remaining_work)
        if best_gpu is None:
            # GPUプールが空の場合はタスクを未完了のまま放置
            return
        
        task.assigned_gpu = best_gpu

        # 自分のGPUを選ぶ場合、ゲストが走っていればプリエンプト
        if self.gpu_owner[best_gpu.gpu_id] == user_id:
            # プリエンプトが発生する場合、このタスクがプリエンプトしたことを記録
            if best_gpu.current_task is not None and best_gpu.current_task.user_id != user_id:
                task.preempted_others_count += 1
            
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

    def process_gpu_finish(self, data):
        # dataは(gpu_id, task_id)のタプル
        gpu_id, expected_task_id = data
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
        
        # タスクIDが一致しない場合は古いイベント（プリエンプトされた）なので無視
        if task.task_id != expected_task_id:
            return
        
        # タスク完了
        task.completion_time = self.current_time
        task.remaining_work = 0.0
        task.completed = True  # 明示的に完了フラグを設定
        gpu.current_task = None

        # 次があれば開始
        if len(gpu.task_queue) > 0:
            next_task = gpu.task_queue.pop(0)
            self.start_task_on_gpu(gpu, next_task)

    # ---------------------- 実行ループ ----------------------
    def run(self):
        self.initialize()
        # 到着は3600秒まで、処理はキューが空になるまで継続
        while self.event_queue:
            time, event_type, data = heapq.heappop(self.event_queue)
            self.current_time = time
            if event_type == "task_arrival":
                self.process_task_arrival(data)
            elif event_type == "gpu_finish":
                self.process_gpu_finish(data)

        # シミュレーション終了時：キューに残っているタスク（完了していないもの）を失敗扱い
        for gpu in self.shared_gpus:
            if gpu.current_task is not None:
                gpu.current_task.failed = True
                gpu.current_task.completion_time = None
            for t in gpu.task_queue:
                if not getattr(t, 'completed', False):
                    t.failed = True
                    t.completion_time = None

        print(f"シミュレーション終了：時刻 {self.current_time}")
        print(f"発生したタスク総数：{len(self.all_tasks)}")
        print(f"\n[プリエンプト統計]")
        print(f"プリエンプト発生回数：{self.preemption_count}")
        print(f"再割当：自分GPU={self.realloc_own}, 元GPU待ち={self.realloc_wait_here}, 他GPU={self.realloc_other}")
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
