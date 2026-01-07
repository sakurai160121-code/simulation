"""
シミュレーション実行（所有者優先度あり割り込み）
複数ユーザーが共有GPUプール内のすべてのGPUを使用
タスク発生時に、自分のGPUは割り込み可能、他人のGPUは実効性能低下を考慮して選択
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


class SimulatorWithOwnerPriority:
    """
    所有者優先度ありの共有GPU版シミュレータ
    所有者のタスクは割り込み可能、他人のGPUは実効性能が低下
    """
    def __init__(self, task_patterns=None):
        self.users = []
        self.shared_gpus = []  # 共有GPUプール
        self.gpu_owner = {}    # GPU ID → ユーザーID（所有者）のマッピング
        self.event_queue = []  # (時刻, イベント種別, データ)
        self.current_time = 0.0
        self.all_tasks = []  # シミュレーション中に発生したすべてのタスク
        self.task_patterns = task_patterns or {}  # タスク発生パターン
        
    def initialize(self):
        """ユーザーと共有GPUプールを初期化"""
        # 共有GPUプール作成（各ユーザーのGPU 20台を共有プール化）
        for user_id in range(NUM_USERS):
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
            self.gpu_owner[user_id] = user_id  # 所有者を記録
        
        # ユーザー作成（GPUは割り当てない、共有プールを使う）
        for user_id in range(NUM_USERS):
            arrival_rate = ARRIVAL_RATES.get(str(user_id), ARRIVAL_RATE)
            user = User(user_id=user_id, gpu=None, arrival_rate=arrival_rate)
            self.users.append(user)
            
            # 最初のタスク発生イベントをスケジュール
            arrivals = self.task_patterns.get("arrivals", {}).get(str(user_id), [])
            if arrivals:
                self.schedule_event(arrivals[0], "task_arrival", user_id)
            else:
                first_arrival = np.random.exponential(1.0 / arrival_rate)
                self.schedule_event(first_arrival, "task_arrival", user_id)
    
    def schedule_event(self, time, event_type, data):
        """イベントをスケジュール"""
        heapq.heappush(self.event_queue, (time, event_type, data))
    
    def get_owner_utilization(self, gpu):
        """
        GPU所有者の稼働率を計算
        ρ_own = λ_own · s̄_own / μ
        """
        owner_id = self.gpu_owner[gpu.gpu_id]
        owner_lambda = ARRIVAL_RATES.get(str(owner_id), ARRIVAL_RATE)
        owner_task_size_mean = TASK_SIZE_MEANS.get(str(owner_id), TASK_SIZE_MEAN)
        
        utilization = owner_lambda * owner_task_size_mean / gpu.processing_rate
        return utilization
    
    def get_effective_processing_rate(self, gpu, user_id):
        """
        ユーザーが GPU を使用する場合の実効処理レートを計算
        自分のGPU: 通常の性能
        他人のGPU: μ_eff = μ / (1 + ρ_own)
        """
        if self.gpu_owner[gpu.gpu_id] == user_id:
            # 自分のGPU：割り込み可能なので通常性能
            return gpu.processing_rate
        else:
            # 他人のGPU：実効性能が低下
            utilization = self.get_owner_utilization(gpu)
            effective_rate = gpu.processing_rate / (1.0 + utilization)
            return effective_rate
    
    def predict_completion_time_own_gpu(self, gpu, user_id):
        """
        自分のGPUでの予想完了時刻
        = max(実行中タスク残り時間, 0) + 自分のキュー内タスク処理時間
        """
        if gpu.current_task is None and len(gpu.task_queue) == 0:
            return self.current_time
        
        # 実行中タスクの残り時間
        current_remaining = 0
        if gpu.current_task is not None:
            current_remaining = max(0, gpu.finish_time - self.current_time)
        
        # 自分のキュー内タスク処理時間
        owner_id = self.gpu_owner[gpu.gpu_id]
        user_queue_time = 0
        for task in gpu.task_queue:
            if task.user_id == user_id:
                # 自分のタスクのサービス時間
                job_sizes = self.task_patterns.get("job_sizes", {}).get(str(task.user_id), {})
                job_size = job_sizes.get(str(task.arrival_time))
                if job_size is None:
                    user_mean = TASK_SIZE_MEANS.get(str(task.user_id), TASK_SIZE_MEAN)
                    job_size = np.random.exponential(user_mean)
                user_queue_time += job_size / gpu.processing_rate
        
        return self.current_time + current_remaining + user_queue_time
    
    def predict_completion_time_other_gpu(self, gpu, user_id):
        """
        他人のGPUでの予想完了時刻
        実効処理レートを使ってキューを概算
        """
        if gpu.current_task is None and len(gpu.task_queue) == 0:
            return self.current_time
        
        completion_time = gpu.finish_time if gpu.current_task is not None else self.current_time
        
        queue_length = len(gpu.task_queue)
        if queue_length > 0:
            effective_rate = self.get_effective_processing_rate(gpu, user_id)
            completion_time += (TASK_SIZE_MEAN_GLOBAL / effective_rate) * queue_length
        
        return completion_time
    
    def select_best_gpu(self, user_id):
        """
        ユーザーにとって最適なGPUを選択
        自分のGPU：割り込み可能
        他人のGPU：実効性能を考慮
        """
        best_gpu = None
        earliest_time = float('inf')
        
        for gpu in self.shared_gpus:
            if self.gpu_owner[gpu.gpu_id] == user_id:
                # 自分のGPU
                completion_time = self.predict_completion_time_own_gpu(gpu, user_id)
            else:
                # 他人のGPU
                completion_time = self.predict_completion_time_other_gpu(gpu, user_id)
            
            if completion_time < earliest_time:
                earliest_time = completion_time
                best_gpu = gpu
        
        return best_gpu
    
    def process_task_arrival(self, user_id):
        """タスク到着イベント処理"""
        user = self.users[user_id]
        task = user.create_task(self.current_time)
        self.all_tasks.append(task)
        
        # 最適なGPUを選択
        best_gpu = self.select_best_gpu(user_id)
        task.assigned_gpu = best_gpu
        # GPU所有者IDを渡して、所有者のタスクを優先化
        owner_id = self.gpu_owner[best_gpu.gpu_id]
        best_gpu.add_task(task, owner_id=owner_id)
        
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
        task.start_time = self.current_time
        gpu.current_task = task
        
        # 処理時間を計算
        job_sizes = self.task_patterns.get("job_sizes", {}).get(str(task.user_id), {})
        job_size = job_sizes.get(str(task.arrival_time))
        if job_size is None:
            user_mean = TASK_SIZE_MEANS.get(str(task.user_id), TASK_SIZE_MEAN)
            job_size = np.random.exponential(user_mean)
        
        # 実際の処理レート（所有者でない場合は実効性能は適用しない、実行開始時点では割り込まれていないため）
        service_time = job_size / gpu.processing_rate
        
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
        task.completion_time = self.current_time
        gpu.current_task = None
        
        # キューに次のタスクがあれば処理開始
        if len(gpu.task_queue) > 0:
            next_task = gpu.task_queue.pop(0)
            self.start_task_on_gpu(gpu, next_task)
    
    def run(self):
        """シミュレーション実行"""
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
    """メイン処理"""
    # タスクパターンを生成（存在しない場合）または読み込み
    if not os.path.exists("task_patterns.json"):
        print("タスクパターンを生成中...")
        save_patterns()
    
    patterns = load_patterns()
    
    # シミュレーション実行
    sim = SimulatorWithOwnerPriority(task_patterns=patterns)
    tasks = sim.run()
    
    # 結果分析と出力
    analyze_and_print_results(tasks, NUM_USERS, mode="with_sharing_owner_priority")


if __name__ == "__main__":
    main()
