"""
シミュレーション実行（共有しないケース）
各ユーザーが自分のGPUのみを使用
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
)
from results import analyze_and_print_results
from task_patterns import load_patterns, save_patterns
import os

np.random.seed(RANDOM_SEED)


class Simulator:
    """
    シミュレータ基底クラス
    """
    def __init__(self, task_patterns=None):
        self.users = []
        self.event_queue = []  # (時刻, イベント種別, データ)
        self.current_time = 0.0
        self.all_tasks = []  # シミュレーション中に発生したすべてのタスク
        self.task_patterns = task_patterns or {}  # タスク発生パターン
        
    def initialize(self):
        """ユーザーとGPUを初期化"""
        for user_id in range(NUM_USERS):
            # ユーザーの性能ティアを決定
            tier = None
            for tier_name, user_list in GPU_TIER_ASSIGNMENT.items():
                if user_id in user_list:
                    tier = tier_name
                    break
            
            # 性能ティアに対応する処理レートを取得
            processing_rate = GPU_PERFORMANCE_LEVELS[tier]
            arrival_rate = ARRIVAL_RATES.get(str(user_id), ARRIVAL_RATE)

            # GPU と User を作成
            gpu = GPU(gpu_id=user_id, processing_rate=processing_rate)
            user = User(user_id=user_id, gpu=gpu, arrival_rate=arrival_rate)
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
    
    def process_task_arrival(self, user_id):
        """タスク到着イベント処理"""
        user = self.users[user_id]
        task = user.create_task(self.current_time)
        self.all_tasks.append(task)
        
        # タスクをユーザーのGPUに割り当て
        task.assigned_gpu = user.gpu
        
        # GPUが空いていたら即座に処理開始、そうでなければキューに追加
        if user.gpu.current_task is None:
            self.start_task_on_gpu(user.gpu, task)
        else:
            user.gpu.add_task(task)
        
        # 次のタスク発生をスケジュール（パターンから取得）
        arrivals = self.task_patterns.get("arrivals", {}).get(str(user_id), [])
        next_arrival_index = user.task_count  # 直接的にtask_countを使用
        
        if next_arrival_index < len(arrivals):
            next_arrival = arrivals[next_arrival_index]
            if next_arrival <= SIMULATION_TIME:
                self.schedule_event(next_arrival, "task_arrival", user_id)
    
    def start_task_on_gpu(self, gpu, task):
        """GPUでタスクを開始"""
        task.start_time = self.current_time
        gpu.current_task = task
        
        # タスクサイズをパターンから取得し、サービス時間を算出
        sizes = self.task_patterns.get("sizes", {}).get(str(task.user_id), {})
        job_size = sizes.get(str(task.arrival_time))
        if job_size is None:
            base_size = TASK_SIZE_MEANS.get(task.user_id, TASK_SIZE_MEAN_GLOBAL)
            batch_size = BATCH_SIZES.get(task.user_id, 1000)
            epochs = EPOCHS.get(task.user_id, 1)
            user_mean = base_size * batch_size * epochs
            job_size = np.random.exponential(user_mean)

        # 合計仕事量を保持
        task.total_work = job_size

        service_time = job_size / gpu.processing_rate
        
        finish_time = self.current_time + service_time
        gpu.finish_time = finish_time
        
        # タスク完了イベントをスケジュール
        self.schedule_event(finish_time, "gpu_finish", gpu.gpu_id)
    
    def process_gpu_finish(self, gpu_id):
        """GPU処理完了イベント処理"""
        gpu = self.users[gpu_id].gpu
        
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
    sim = Simulator(task_patterns=patterns)
    tasks = sim.run()
    
    # 結果分析と出力
    analyze_and_print_results(tasks, NUM_USERS, mode="no_sharing")


if __name__ == "__main__":
    main()
