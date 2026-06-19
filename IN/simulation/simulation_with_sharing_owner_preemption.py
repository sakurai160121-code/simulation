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
import config
from definitions import User, GPU, Task
from config import (
    NUM_USERS,
    ARRIVAL_RATE,
    SIMULATION_TIME,
    RANDOM_SEED,
    GPU_PERFORMANCE_LEVELS,
    GPU_TIER_ASSIGNMENT,
    EXPECTED_TASK_SIZE,
    get_user_expected_task_size,
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
        self.job_size_none_count = 0
        self.job_size_fallback_count = 0

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

        # ACP常駐GPUを共有プールへ追加（所有者なし）
        for acp_spec in config.get_acp_resident_gpu_specs():
            gpu = GPU(
                gpu_id=acp_spec["gpu_id"],
                processing_rate=acp_spec["processing_rate"],
            )
            self.shared_gpus.append(gpu)
            self.gpu_owner[gpu.gpu_id] = acp_spec.get("owner")

        # ユーザー作成（共有プール運用）
        for user_id in range(NUM_USERS):
            arrival_rate = config.ARRIVAL_RATES.get(str(user_id), ARRIVAL_RATE)
            user = User(user_id=user_id, gpu=None, arrival_rate=arrival_rate)
            self.users.append(user)

            arrivals = self.task_patterns.get("arrivals", {}).get(str(user_id), [])
            if arrivals:
                self.schedule_event(arrivals[0], "task_arrival", user_id)
            elif arrival_rate > 0:
                first_arrival = np.random.exponential(1.0 / arrival_rate)
                self.schedule_event(first_arrival, "task_arrival", user_id)

    def schedule_event(self, time, event_type, data):
        heapq.heappush(self.event_queue, (time, event_type, data))

    def resolve_task_profile(self, user_id, task_index):
        """到着順インデックスで task_type と job_size を解決する。"""
        user_id_str = str(user_id)
        task_type = "inference"
        job_size = None

        user_types = self.task_patterns.get("types", {}).get(user_id_str, {})
        user_sizes = self.task_patterns.get("sizes", {}).get(user_id_str, {})
        type_values = list(user_types.values())
        size_values = list(user_sizes.values())

        if task_index < len(type_values):
            task_type = type_values[task_index]
        if task_index < len(size_values):
            job_size = size_values[task_index]

        if job_size is None:
            self.job_size_none_count += 1
            job_size = EXPECTED_TASK_SIZE.get(task_type, EXPECTED_TASK_SIZE["inference"])
            self.job_size_fallback_count += 1

        return task_type, float(job_size)

    # ---------------------- 実効性能・待ち時間推定 ----------------------
    def get_owner_utilization(self, gpu):
        owner_id = self.gpu_owner.get(gpu.gpu_id)
        if owner_id is None:
            return 0.0
        owner_lambda = config.ARRIVAL_RATES.get(str(owner_id), ARRIVAL_RATE)
        owner_task_size_mean = get_user_expected_task_size(owner_id)
        return owner_lambda * owner_task_size_mean / gpu.processing_rate

    def get_effective_processing_rate(self, gpu, user_id):
        owner_id = self.gpu_owner.get(gpu.gpu_id)
        if owner_id is None or owner_id == user_id:
            return gpu.processing_rate
        else:
            rho_own = self.get_owner_utilization(gpu)
            return gpu.processing_rate / (1.0 + rho_own)

    def predict_owner_wait_on_gpu(self, gpu, owner_id):
        if owner_id is None:
            return 0.0
        # 実行中が所有者ならその残りを待つ、ゲストなら待たない
        wait = 0.0
        if gpu.current_task is not None:
            if gpu.current_task.user_id == owner_id:
                wait += max(0.0, gpu.finish_time - self.current_time)
        for t in gpu.task_queue:
            if t.user_id == owner_id:
                # 所有者キュー分
                size = EXPECTED_TASK_SIZE.get(t.task_type, EXPECTED_TASK_SIZE["inference"])
                wait += size / gpu.processing_rate
        return wait

    def expected_interruption_penalty(self, gpu, service_time, task_type="inference"):
        # 所有者到来率による途中切断リスクの期待ペナルティ
        owner_id = self.gpu_owner.get(gpu.gpu_id)
        if owner_id is None:
            return 0.0
        lam = config.ARRIVAL_RATES.get(str(owner_id), ARRIVAL_RATE)
        mean_owner_size = get_user_expected_task_size(owner_id)
        mean_owner_service = mean_owner_size / gpu.processing_rate
        # 期待割込み回数（Poissonの期待値）：lam * service_time を用いる強化版
        expected_interruptions = lam * service_time
        # タスク種別に応じたオーバーヘッド係数を選択（config モジュール経由で参照して動的更新を反映）
        if task_type == "training":
            overhead_factor = config.INTERRUPTION_OVERHEAD_FACTOR_TRAINING
        else:
            overhead_factor = config.INTERRUPTION_OVERHEAD_FACTOR_INFERENCE
        # 追加の再開・マイグレーションオーバーヘッドを係数で加味（強めに設定）
        penalty = expected_interruptions * (mean_owner_service * (1.0 + overhead_factor))
        return penalty

    # ---------------------- GPU選択ロジック ----------------------
    def get_interruption_overhead_factor(self, task_type="inference"):
        """プリエンプト時に残作業へ加える係数を返す。"""
        if task_type == "training":
            return config.INTERRUPTION_OVERHEAD_FACTOR_TRAINING
        return config.INTERRUPTION_OVERHEAD_FACTOR_INFERENCE
    def _get_prediction_work(self, user_id, remaining_work=None, new_task_type=None):
        """予測に使う仕事量を返す。新規到着時は task_type、再開時は remaining_work を使う。"""
        if new_task_type is not None:
            return EXPECTED_TASK_SIZE.get(new_task_type, EXPECTED_TASK_SIZE["inference"])
        if remaining_work is not None:
            return remaining_work
        return get_user_expected_task_size(user_id)

    def predict_completion_time_own_gpu(self, gpu, user_id, remaining_work=None, new_task_type=None):
        # 新規到着の選択では、実際に走っている先行タスクを必ず待つ前提で見積もる。
        # これにより、プリエンプトが実際に発生しないケースでの過度な楽観見積もりを防ぐ。
        wait_owner = 0.0
        if gpu.current_task is not None:
            wait_owner += max(0.0, gpu.finish_time - self.current_time)
        for queued_task in gpu.task_queue:
            if queued_task.user_id == user_id:
                size = EXPECTED_TASK_SIZE.get(queued_task.task_type, EXPECTED_TASK_SIZE["inference"])
                wait_owner += size / gpu.processing_rate
        work = self._get_prediction_work(user_id, remaining_work=remaining_work, new_task_type=new_task_type)
        return self.current_time + wait_owner + work / gpu.processing_rate

    def predict_completion_time_other_gpu(self, gpu, user_id, remaining_work=None, new_task_type=None, include_penalty=True):
        base = gpu.finish_time if gpu.current_task is not None else self.current_time
        # 既存キューの正確な計算（各タスクの実際のサイズを使用、バッチ係数適用）
        mu_eff = self.get_effective_processing_rate(gpu, user_id)
        queue_time = 0.0
        for task in gpu.task_queue:
            task_size_mean = EXPECTED_TASK_SIZE.get(task.task_type, EXPECTED_TASK_SIZE["inference"])
            queue_time += task_size_mean / mu_eff
        work = self._get_prediction_work(user_id, remaining_work=remaining_work, new_task_type=new_task_type)
        service_time = work / mu_eff
        penalty = 0.0
        if include_penalty:
            # タスク種別を指定してペナルティ計算
            task_type = new_task_type if new_task_type is not None else "inference"
            penalty = self.expected_interruption_penalty(gpu, service_time, task_type=task_type)
        return base + queue_time + service_time + penalty

    def select_best_gpu_for_new(self, user_id, remaining_work, new_task_type):
        if not self.shared_gpus:
            return None
        
        best_gpu = None
        best_time = float('inf')
        for gpu in self.shared_gpus:
            if self.gpu_owner.get(gpu.gpu_id) == user_id:
                t = self.predict_completion_time_own_gpu(
                    gpu,
                    user_id,
                    remaining_work=remaining_work,
                    new_task_type=new_task_type,
                )
            else:
                t = self.predict_completion_time_other_gpu(
                    gpu,
                    user_id,
                    remaining_work=remaining_work,
                    new_task_type=new_task_type,
                    include_penalty=False,
                )
            if t < best_time:
                best_time = t
                best_gpu = gpu
        return best_gpu

    def select_after_preempt(self, task, preempt_gpu):
        # 1) 自分のGPU（参加していない場合はスキップ）
        own_gpu = None
        for gpu in self.shared_gpus:
            if self.gpu_owner.get(gpu.gpu_id) == task.user_id:
                own_gpu = gpu
                break
        
        if own_gpu is None:
            # 自分のGPUがない場合は他の選択肢のみを検討
            t_own = float('inf')
        else:
            t_own = self.predict_completion_time_own_gpu(own_gpu, task.user_id, remaining_work=task.remaining_work)

        # 2) プリエンプト元GPUで先頭待機（所有者待ち）
        wait_owner = self.predict_owner_wait_on_gpu(preempt_gpu, self.gpu_owner.get(preempt_gpu.gpu_id))
        # ゲストはμ_effで処理されるためサービス率を実効値に
        mu_eff_here = self.get_effective_processing_rate(preempt_gpu, task.user_id)
        service_time_here = task.remaining_work / mu_eff_here
        # 再開後の割り込みリスクを考慮（タスク種別を指定）
        penalty_here = self.expected_interruption_penalty(preempt_gpu, service_time_here, task_type=task.task_type)
        t_wait_here = self.current_time + wait_owner + service_time_here + penalty_here

        # 3) 他GPUのキュー末尾（期待ペナルティ込み）から最良を探す
        best_other_time = float('inf')
        best_other_gpu = None
        for gpu in self.shared_gpus:
            if gpu is preempt_gpu:
                continue
            t = self.predict_completion_time_other_gpu(gpu, task.user_id, remaining_work=task.remaining_work)
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

        if task in gpu.task_queue:
            gpu.task_queue.remove(task)
        
        # 初回のみ最初の実行開始時刻を記録（待ち時間計測用）
        if task.first_execution_start_time is None:
            task.first_execution_start_time = self.current_time
        # 最後の実行開始時刻は常に更新（中断再開時に対応）
        task.last_execution_start_time = self.current_time
        task.start_time = self.current_time  # 後方互換性
        gpu.current_task = task
        # 実行時のサービス速度は owner-priority と同じく実GPU性能を用いる。
        # プリエンプティブ方式の差分は「割り込みの可否」に限定する。
        rate = gpu.processing_rate
        service_time = task.remaining_work / rate
        finish_time = self.current_time + service_time
        gpu.finish_time = finish_time
        # イベントデータにGPU IDとタスクIDを含める
        self.schedule_event(finish_time, "gpu_finish", (gpu.gpu_id, task.task_id))

    def preempt_guest_if_needed(self, gpu, owner_id):
        if owner_id is None:
            return
        if gpu.current_task is not None and gpu.current_task.user_id != owner_id:
            # ゲストをプリエンプト
            self.preemption_count += 1
            guest = gpu.current_task
            
            # プリエンプト回数を記録
            guest.preempted_count += 1
            
            # このGPUでの実行時間を累積（重要：プリエンプション対応）
            elapsed = max(0.0, self.current_time - (guest.last_execution_start_time or self.current_time))
            guest.accumulated_service_time += elapsed
            
            rate_used = gpu.processing_rate
            processed_work = elapsed * rate_used
            # 初回実行時に残作業が未設定なら設定
            if getattr(guest, 'remaining_work', None) is None:
                guest.remaining_work = guest.job_size
                if guest.remaining_work is None:
                    self.job_size_none_count += 1
                    guest.remaining_work = EXPECTED_TASK_SIZE.get(guest.task_type, EXPECTED_TASK_SIZE["inference"])
                    guest.job_size = guest.remaining_work
                    self.job_size_fallback_count += 1
            overhead_factor = self.get_interruption_overhead_factor(guest.task_type)
            # プリエンプトのたびに、その分の再開コストを追加する
            guest.remaining_work += processed_work * overhead_factor
            guest.remaining_work = max(0.0, guest.remaining_work - processed_work)

            # プリエンプト状態へ：GPUから降ろす
            gpu.current_task = None

            # ゲストの次の行き先を決める
            best_time, choice, dest_gpu = self.select_after_preempt(guest, gpu)
            if choice == 'own':
                self.realloc_own += 1
                dest_gpu.add_task(guest, owner_id=guest.user_id)
                guest.assigned_gpu = dest_gpu
                guest.assigned_time = self.current_time  # 再割当時刻
                if dest_gpu.current_task is None:
                    self.start_task_on_gpu(dest_gpu, guest)
            elif choice == 'wait_here':
                self.realloc_wait_here += 1
                gpu.task_queue.insert(0, guest)  # 先頭で待機
                guest.assigned_gpu = gpu
                guest.assigned_time = self.current_time  # 再割当時刻
            else:  # other
                self.realloc_other += 1
                dest_gpu.add_task(guest)  # 末尾
                guest.assigned_gpu = dest_gpu
                guest.assigned_time = self.current_time  # 再割当時刻
                if dest_gpu.current_task is None:
                    self.start_task_on_gpu(dest_gpu, guest)

    # ---------------------- イベント処理 ----------------------
    def process_task_arrival(self, user_id):
        user = self.users[user_id]
        task_index = user.task_count
        task_type, job_size = self.resolve_task_profile(user_id, task_index)
        task = user.create_task(self.current_time, task_type=task_type)
        # タスクサイズ→残作業として保持
        task.job_size = job_size
        task.remaining_work = job_size
        task.total_work = job_size
        self.all_tasks.append(task)

        # 最適GPU選択（他GPUは中断リスク期待値込み）
        best_gpu = self.select_best_gpu_for_new(user_id, task.remaining_work, task_type)
        if best_gpu is None:
            # GPUプールが空の場合はタスクを未完了のまま放置
            return
        
        task.assigned_gpu = best_gpu
        task.assigned_time = self.current_time  # GPU割り当て時刻

        # 自分のGPUを選ぶ場合、ゲストが走っていればプリエンプト
        if self.gpu_owner.get(best_gpu.gpu_id) == user_id:
            # プリエンプトが発生する場合、このタスクがプリエンプトしたことを記録
            if best_gpu.current_task is not None and best_gpu.current_task.user_id != user_id:
                task.preempted_others_count += 1
            
            self.preempt_guest_if_needed(best_gpu, owner_id=user_id)

        # キューへ投入（所有者優先）
        best_gpu.add_task(task, owner_id=self.gpu_owner.get(best_gpu.gpu_id))

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
        
        # タスク完了時に、最後の実行区間を累積サービス時間に加算
        if task.last_execution_start_time is not None:
            elapsed = max(0.0, self.current_time - task.last_execution_start_time)
            task.accumulated_service_time += elapsed
        
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
