import wx
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar
)


class PopFrame(wx.Frame):
    def __init__(self, title, fig):
        super().__init__(None, title=title, size=(1000, 1000))
        #fig.set_dpi(150)
        cv = FigureCanvas(self, -1, fig._fig)

        nav_toolbar = NavigationToolbar(cv)
        nav_toolbar.Realize()
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(nav_toolbar, 0, wx.ALIGN_CENTER)
        sizer.Add(cv, 1, wx.EXPAND)
        nav_toolbar.update()

        self.SetSizer(sizer)

        cv.Show()
        cv.draw_idle()
        cv.flush_events()

