"""
シミュレーション実行（共有しないケース）
各ユーザーが自分のGPUのみを使用
"""

import numpy as np
import heapq
from definitions import User, GPU, Task
from config import NUM_USERS, ARRIVAL_RATE, SIMULATION_TIME, RANDOM_SEED, GPU_PERFORMANCE_LEVELS, GPU_TIER_ASSIGNMENT
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
            
            # 性能ティアに対応する処理時間の平均を取得
            mean_processing_time = GPU_PERFORMANCE_LEVELS[tier]
            
            # GPU と User を作成
            gpu = GPU(gpu_id=user_id, processing_time=mean_processing_time)
            user = User(user_id=user_id, gpu=gpu, arrival_rate=ARRIVAL_RATE)
            self.users.append(user)
            
            # 最初のタスク発生イベントをスケジュール
            first_arrival = np.random.exponential(1.0 / ARRIVAL_RATE)
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
        user.gpu.add_task(task)
        
        # GPUが空いていたら即座に処理開始
        if user.gpu.current_task is None:
            self.start_task_on_gpu(user.gpu, task)
        
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
        
        # 処理時間をパターンから取得
        service_times = self.task_patterns.get("service_times", {}).get(str(gpu.gpu_id), {})
        service_time = service_times.get(str(task.arrival_time), np.random.exponential(gpu.processing_time))
        
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
    sim = Simulator(task_patterns=patterns)
    tasks = sim.run()
    
    # 結果分析と出力
    analyze_and_print_results(tasks, NUM_USERS, mode="no_sharing")


if __name__ == "__main__":
    main()
