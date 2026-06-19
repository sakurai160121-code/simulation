"""
シミュレーション実行（ACPで共有：全員参加、プリエンプティブ方式なし）
複数ユーザーが共有GPUプール内のすべてのGPUを使用
タスク発生時に完了時刻が最も早いGPUを選択
"""


import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
import numpy as np
import heapq
from simulation.core import config
from simulation.core.definitions import User, GPU, Task
from simulation.core.config import (
    NUM_USERS,
    ARRIVAL_RATE,
    SIMULATION_TIME,
    RANDOM_SEED,
    GPU_PERFORMANCE_LEVELS,
    GPU_TIER_ASSIGNMENT,
    EXPECTED_TASK_SIZE,
)
from simulation.analysis.results import analyze_and_print_results
from simulation.engine.task_patterns import load_patterns, save_patterns
import os

np.random.seed(RANDOM_SEED)


class SimulatorWithSharing:
    """
    共有GPU版シミュレータ
    複数ユーザーが共有GPUプールを使用
    """
    def __init__(self, task_patterns=None, participating_users=None):
        self.users = []
        self.shared_gpus = []  # 共有GPUプール
        self.event_queue = []  # (時刻, イベント種別, データ)
        self.current_time = 0.0
        self.all_tasks = []  # シミュレーション中に発生したすべてのタスク
        self.task_patterns = task_patterns or {}  # タスク発生パターン
        self.participating_users = participating_users if participating_users is not None else list(range(NUM_USERS))
        self.job_size_none_count = 0
        self.job_size_fallback_count = 0
        
    def initialize(self):
        """ユーザーと共有GPUプールを初期化"""
        # 共有GPUプール作成（参加ユーザーのGPUのみを共有プール化）
        for user_id in self.participating_users:
            # ユーザーの性能ティアを決定
            tier = None
            for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
                if user_id in user_list:
                    tier = tier_name
                    break
            
            # 性能ティアに対応する処理レートを取得
            processing_rate = GPU_PERFORMANCE_LEVELS[tier]
            
            # GPU を共有プールに追加
            gpu = GPU(gpu_id=user_id, processing_rate=processing_rate)
            self.shared_gpus.append(gpu)

        # ACP常駐GPUを共有プールへ追加
        for acp_spec in config.get_acp_resident_gpu_specs():
            gpu = GPU(
                gpu_id=acp_spec["gpu_id"],
                processing_rate=acp_spec["processing_rate"],
            )
            self.shared_gpus.append(gpu)
        
        # ユーザー作成（GPUは割り当てない、共有プールを使う）
        for user_id in range(NUM_USERS):
            arrival_rate = config.ARRIVAL_RATES.get(str(user_id), ARRIVAL_RATE)
            user = User(user_id=user_id, gpu=None, arrival_rate=arrival_rate)
            self.users.append(user)
            
            # 最初のタスク発生イベントをスケジュール（task_patternsから取得）
            arrivals = self.task_patterns.get("arrivals", {}).get(str(user_id), [])
            if arrivals and len(arrivals) > 0:
                first_arrival = arrivals[0]
                if first_arrival <= SIMULATION_TIME:
                    self.schedule_event(first_arrival, "task_arrival", user_id)
    
    def schedule_event(self, time, event_type, data):
        """イベントをスケジュール"""
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
    
    def predict_completion_time(self, gpu, new_task_type):
        """
        GPUの予想完了時刻を計算（正確版）
        キュー内の待ち時間に加えて、新規到着タスク自身の処理時間も含める
        """
        # 現在のタスクが完了する時刻
        completion_time = gpu.finish_time if gpu.current_task is not None else self.current_time
        
        # キュー内タスクの期待処理時間を加算
        for task in gpu.task_queue:
            task_size_mean = EXPECTED_TASK_SIZE.get(task.task_type, EXPECTED_TASK_SIZE["inference"])
            completion_time += task_size_mean / gpu.processing_rate

        # 新規到着タスク自身の期待処理時間を加算
        new_task_size = EXPECTED_TASK_SIZE.get(new_task_type, EXPECTED_TASK_SIZE["inference"])
        completion_time += new_task_size / gpu.processing_rate
        
        return completion_time
    
    def get_service_time(self, gpu, task):
        """タスクサイズを取得し、GPU処理レートでサービス時間を計算"""
        job_size = task.job_size
        if job_size is None:
            self.job_size_none_count += 1
            job_size = EXPECTED_TASK_SIZE.get(task.task_type, EXPECTED_TASK_SIZE["inference"])
            task.job_size = job_size
            self.job_size_fallback_count += 1
        # 合計仕事量を保持
        task.total_work = job_size
        return job_size / gpu.processing_rate
    
    def select_best_gpu(self, new_task_type):
        """
        完了時刻が最も早いGPUを選択
        Returns: GPU
        """
        if not self.shared_gpus:
            return None
        
        best_gpu = None
        earliest_time = float('inf')
        
        for gpu in self.shared_gpus:
            completion_time = self.predict_completion_time(gpu, new_task_type)
            if completion_time < earliest_time:
                earliest_time = completion_time
                best_gpu = gpu
        
        return best_gpu
    
    def process_task_arrival(self, user_id):
        """タスク到着イベント処理"""
        user = self.users[user_id]
        task_index = user.task_count
        task_type, job_size = self.resolve_task_profile(user_id, task_index)
        task = user.create_task(self.current_time, task_type=task_type)
        task.job_size = job_size
        task.total_work = job_size
        self.all_tasks.append(task)
        
        # 最適なGPUを選択（到着タスクのtypeを予測へ反映）
        best_gpu = self.select_best_gpu(task_type)
        if best_gpu is None:
            # GPUプールが空の場合はタスクを未完了のまま放置
            return
        
        task.assigned_gpu = best_gpu
        task.assigned_time = self.current_time  # GPU割り当て時刻
        best_gpu.add_task(task)
        
        # GPUが空いていたら即座に処理開始
        if best_gpu.current_task is None:
            self.start_task_on_gpu(best_gpu, task)
        
        # 次のタスク発生をスケジュール（パターンから取得）
        arrivals = self.task_patterns.get("arrivals", {}).get(str(user_id), [])
        next_arrival_index = len([t for t in user.tasks if t is not None])
        
        if next_arrival_index < len(arrivals):
            next_arrival = arrivals[next_arrival_index]
            if next_arrival <= SIMULATION_TIME:
                self.schedule_event(next_arrival, "task_arrival", user_id)
    
    def start_task_on_gpu(self, gpu, task):
        """GPUでタスクを開始"""
        if task in gpu.task_queue:
            gpu.task_queue.remove(task)
        # 初回のみ最初の実行開始時刻を記録
        if task.first_execution_start_time is None:
            task.first_execution_start_time = self.current_time
        # 最後の実行開始時刻は常に更新（再開時に対応）
        task.last_execution_start_time = self.current_time
        task.start_time = self.current_time  # 後方互換性
        gpu.current_task = task
        
        # 処理時間をパターンから取得
        service_time = self.get_service_time(gpu, task)
        
        finish_time = self.current_time + service_time
        gpu.finish_time = finish_time
        
        # タスク完了イベントをスケジュール
        self.schedule_event(finish_time, "gpu_finish", gpu.gpu_id)
    
    def process_gpu_finish(self, gpu_id):
        """GPU処理完了イベント処理"""
        # GPU IDで対応するGPUを探す
        gpu = None
        for g in self.shared_gpus:
            if g.gpu_id == gpu_id:
                gpu = g
                break
        
        if gpu is None:
            return
        
        # 現在のタスクを完了
        task = gpu.current_task
        
        # 実行区間を累積に加算
        if task.last_execution_start_time is not None:
            elapsed = max(0.0, self.current_time - task.last_execution_start_time)
            task.accumulated_service_time += elapsed
        
        task.completion_time = self.current_time
        gpu.current_task = None
        
        # キューに次のタスクがあれば処理開始
        if len(gpu.task_queue) > 0:
            next_task = gpu.task_queue.pop(0)
            self.start_task_on_gpu(gpu, next_task)
    
    def run(self):
        """シミュレーション実行"""
        self.initialize()
        
        # 到着は3600秒まで、処理はキューが空になるまで継続
        while self.event_queue:
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
    """メイン処理"""
    # タスクパターンを生成（存在しない場合）または読み込み
    if not os.path.exists("task_patterns.json"):
        print("タスクパターンを生成中...")
        save_patterns()
    
    patterns = load_patterns()
    
    # シミュレーション実行
    sim = SimulatorWithSharing(task_patterns=patterns)
    tasks = sim.run()
    
    # 結果分析と出力
    analyze_and_print_results(tasks, NUM_USERS, mode="with_sharing")


if __name__ == "__main__":
    main()
