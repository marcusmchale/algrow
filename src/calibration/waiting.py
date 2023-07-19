import logging

import wx

logger = logging.getLogger(__name__)


class WaitPanel(wx.Panel):
    def __init__(self, parent):
        logger.debug("Launch waiting panel")
        super().__init__(parent)

        self.wait_btn = wx.Button(self, 1, "Please wait...")
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.wait_btn, 0, wx.ALIGN_CENTER)
        self.SetSizer(self.sizer)

        logger.debug("Waiting panel loaded")
