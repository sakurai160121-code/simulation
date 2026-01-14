"""
エンティティの定義
"""

class GPU:
    """
    GPU クラス
    各ユーザーが1台所有
    """
    def __init__(self, gpu_id, processing_rate=1.0):
        self.gpu_id = gpu_id
        self.processing_rate = processing_rate  # 単位タスクサイズあたりの処理レート（大きいほど速い）
        self.task_queue = []  # キューイング中のタスク
        self.current_task = None  # 現在処理中のタスク
        self.finish_time = 0  # 現在のタスク完了予定時刻
        
    def add_task(self, task, owner_id=None):
        """
        タスクをキューに追加
        owner_id が指定されている場合、所有者のタスクを優先
        """
        if owner_id is not None and task.user_id == owner_id:
            # 所有者のタスク：既存の所有者タスクの後、ゲストタスクの前に挿入
            # 所有者タスク間ではFCFSを維持
            insert_pos = 0
            for i, t in enumerate(self.task_queue):
                if t.user_id == owner_id:
                    insert_pos = i + 1
                else:
                    break
            self.task_queue.insert(insert_pos, task)
        else:
            # 他人のタスク：末尾に追加
            self.task_queue.append(task)
        
    def get_queue_length(self):
        """現在のキュー長（処理中のタスクも含む）"""
        queue_len = len(self.task_queue)
        if self.current_task is not None:
            queue_len += 1
        return queue_len


class Task:
    """
    タスク クラス
    """
    def __init__(self, task_id, user_id, arrival_time):
        self.task_id = task_id
        self.user_id = user_id
        self.arrival_time = arrival_time  # タスク発生時刻
        self.start_time = None  # 処理開始時刻
        self.completion_time = None  # 完了時刻
        self.assigned_gpu = None  # 割り当てられたGPU
        self.total_work = None  # タスク全体の仕事量（TFLOPs）
        # 共有・プリエンプトシナリオ用：残作業量（タスクサイズ）。開始時に設定、プリエンプトで更新
        self.remaining_work = None
        # 期限（締切）と失敗フラグ
        self.deadline = None
        self.failed = False
        
    def get_waiting_time(self):
        """待ち時間 = 開始時刻 - 発生時刻"""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time


class User:
    """
    ユーザー クラス
    """
    def __init__(self, user_id, gpu, arrival_rate=1.0):
        self.user_id = user_id
        self.gpu = gpu  # 自分のGPU（1台）
        self.arrival_rate = arrival_rate  # タスク到着率（ポアソン過程のλ）
        self.task_count = 0  # 発生させたタスク総数
        self.tasks = []  # このユーザーが発生させたタスク一覧
        
    def create_task(self, arrival_time):
        """新しいタスクを生成"""
        task = Task(
            task_id=f"user{self.user_id}_task{self.task_count}",
            user_id=self.user_id,
            arrival_time=arrival_time
        )
        self.task_count += 1
        self.tasks.append(task)
        return task
