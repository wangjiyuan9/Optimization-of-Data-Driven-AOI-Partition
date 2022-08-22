import sys
from update_thread import UpdateThread
from PySide6.QtWidgets import QApplication
from gridworld.grid_world import GridWorld
from pg_learning import PolicyGradientRL

if __name__ == '__main__':
    app = QApplication([])
    world = GridWorld()

    world.data_deputy.read_aoi("./data/AOI_20_grid.npy")
    # world.data_deputy.read_aoi("./data/2.jpg")
    # world.data_deputy.read_traces("./data/trace/trace_1.npy")
    # world.data_deputy.read_parcels("./data/parcels_n.npy")

    indexes = []
    bias = 1
    world.render_deputy.trace_indexes = indexes
    world.render_deputy.parcels_indexes = indexes

    """配置aoi强化学习"""
    aoi_learning = PolicyGradientRL()

    """绑定并启动aoi学习线程"""
    aoi_thread = UpdateThread(runner=aoi_learning.execute, slot=world.aoi_update)
    aoi_thread.start()

    """主进程进入事件循环"""
    world.show()
    sys.exit(app.exec())
