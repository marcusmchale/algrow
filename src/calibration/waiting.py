import logging

import wx
from pubsub import pub

logger = logging.getLogger(__name__)


class WaitPanel(wx.Panel):
    def __init__(self, parent):
        logger.debug("Launch waiting panel")
        super().__init__(parent)

        self.wait_btn = wx.Button(self, 1, "Please wait...")
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        #self.SetSizer(self.sizer)
        #self.toolbar = wx.ToolBar(self, id=-1, style=wx.TB_HORIZONTAL | wx.TB_TEXT)  # | wx.TB_TEXT)

        # start again button
        #self.wait_btn = wx.Button(self.toolbar, 1, "Please wait...segmenting images")

        #self.toolbar.AddControl(self.wait_btn)
        self.sizer.Add(self.wait_btn, 0, wx.ALIGN_CENTER)
        #self.toolbar.Realize()
        self.SetSizer(self.sizer)

        logger.debug("Waiting panel loaded")
