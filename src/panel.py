from open3d.visualization import gui

from collections import defaultdict
from typing import Callable, Type, Union, Optional, Dict, Tuple, List

from .options import DebugEnum

import logging

logger = logging.getLogger(__name__)


class Panel:

    DEFAULTS = {
        str: "",
        float: 0.0,
        int: 1,
        gui.ColorEdit: gui.Color(1.0, 1.0, 1.0),
        gui.Checkbox: True
    }

    def __init__(
            self,
            layout: Union[gui.Layout1D, gui.LayoutContext],
            parent: Union[gui.Window, gui.Layout1D, 'Panel'],
            spacing=0,
            margin=0
    ):
        self.layout = layout
        self.parent = parent
        self.spacing = 0
        self.margin = 0

        if isinstance(self.parent, Panel):
            self.inputs = self.parent.inputs
            self.inputs_to_types = self.parent.inputs_to_types
            self.buttons = self.parent.buttons
            self.checkboxes = self.parent.checkboxes
            self.button_pools = self.parent.button_pools
            self.parent.add_child(self)
            self.list_views = self.parent.list_views
            self.list_view_lists = self.parent.list_view_lists
            self.list_view_callbacks = self.parent.list_view_callbacks
            self.path_select_labels = self.parent.path_select_labels
            self.comboboxes = self.parent.comboboxes
        else:
            self.inputs = dict()
            self.inputs_to_types = dict()
            self.buttons = dict()
            self.checkboxes = dict()
            self.button_pools = dict()
            self.parent.add_child(self.layout)
            self.list_views = dict()
            self.list_view_lists = defaultdict(list)
            self.list_view_callbacks = defaultdict(callable)
            self.path_select_labels = defaultdict(lambda: gui.Label(""))
            self.comboboxes = dict()

        self.children = list()

    @property
    def visible(self):
        return self.layout.visible

    @visible.setter
    def visible(self, visible: bool):
        self.layout.visible = visible

    def add_child(self, panel: 'Panel'):
        self.children.append(panel)
        self.layout.add_child(panel.layout)

    def add_stretch(self):
        self.layout.add_stretch()

    def add_label(self, label: str, font_style=None, font_colour: gui.Color = None):
        label = gui.Label(label)
        if font_style is not None:
            logger.debug(f"font_style: {font_style}")
            label.font_id = font_style
        if font_colour is not None:
            label.text_color = font_colour
        self.layout.add_child(label)

    def add_button(self, key: str, on_clicked: Callable = None, tooltip: Optional[str] = None, toggleable=False, enabled=True):
        button_layout = gui.Horiz()
        button = gui.Button(key)
        button_layout.add_child(button)
        button_layout.add_stretch()
        if on_clicked is not None:
            button.set_on_clicked(on_clicked)
        button.enabled = enabled
        if toggleable:
            button.toggleable = True
            button.is_on = True
        if tooltip is not None:
            button.tooltip = tooltip
        self.buttons[key] = button
        self.inputs_to_types[key] = gui.Button
        self.layout.add_child(button_layout)

    def add_path_select(
            self,
            key: str,
            on_add: Callable,
            on_remove: Callable,
            value=None,
            tooltip: Optional[str] = None,
            enabled=True
    ):
        layout = gui.Horiz()
        button = gui.Button(key)
        if tooltip is not None:
            button.tooltip = tooltip
        button.enabled = enabled

        if value is not None:
            label = gui.Label(value)
        else:
            label = gui.Label("")

        remove = gui.Button("-")
        remove.set_on_clicked(on_remove)
        self.path_select_labels[key] = label
        button.set_on_clicked(on_add)
        layout.add_child(button)
        layout.add_child(remove)
        layout.add_child(label)
        self.layout.add_child(layout)

    def add_checkbox(self, key: str, checked=True, on_checked: Callable = None, tooltip: Optional[str] = None, enabled=True):
        checkbox_layout = gui.Horiz()
        checkbox = gui.Checkbox(key)
        checkbox_layout.add_child(checkbox)
        checkbox_layout.add_stretch()
        if on_checked is not None:
            checkbox.set_on_checked(on_checked)
        if tooltip is not None:
            checkbox.tooltip = tooltip
        checkbox.enabled = enabled
        if checked is not None:
            checkbox.checked = checked
        self.checkboxes[key] = checkbox
        self.inputs_to_types[key] = gui.Checkbox
        self.layout.add_child(checkbox_layout)

    def add_button_pool(self, key):
        layout = gui.Vert()
        self.layout.add_child(layout)
        self.button_pools[key] = ButtonPool(layout)
        layout.add_stretch()

    def add_separation(self, size):
        separation_height = size * self.margin
        self.layout.add_fixed(separation_height)

    def add_input(
            self,
            key: str,
            input_type: Type,  # [str, float, int, gui.Color],
            value: Union[str, float, int, gui.Color, None] = None,
            tooltip: Optional[str] = None,
            on_changed: Optional[Callable] = None,
    ):
        if input_type == str:
            self.inputs[key] = gui.TextEdit()
        elif input_type == float:
            self.inputs[key] = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        elif input_type == int:
            self.inputs[key] = gui.NumberEdit(gui.NumberEdit.INT)
        elif input_type == gui.Color:
            self.inputs[key] = gui.ColorEdit()
        else:
            raise TypeError("Unsupported input type")

        if tooltip is not None:
            self.inputs[key].tooltip = tooltip
        if on_changed is not None:
            self.inputs[key].set_on_value_changed(on_changed)

        self.inputs_to_types[key] = input_type
        self.set_value(key, value)

        # To ensure input is labelled we actually add a new layout for each
        if input_type == gui.Color:
            input_layout = gui.Vert(self.margin, gui.Margins(self.margin))
        else:
            input_layout = gui.Horiz(self.margin, gui.Margins(self.margin))
        input_layout.add_stretch()
        input_layout.add_child(gui.Label(key))
        input_layout.add_child(self.inputs[key])
        self.layout.add_child(input_layout)

    def add_combobox(self, key: str, items: list, callback: Callable, value: DebugEnum):
        layout = gui.Horiz()
        layout.add_child(gui.Label(key))
        combobox = gui.Combobox()
        layout.add_child(combobox)
        self.comboboxes[key] = combobox
        for i in items:
            combobox.add_item(i)
        combobox.set_on_selection_changed(callback)
        combobox.selected_text = value.name
        self.layout.add_child(layout)

    def add_list_view(
            self,
            key: str,
            callback=None,  # to be called whenever value is changed
            n=10
    ):
        vert = gui.Vert(self.margin, gui.Margins(self.margin))
        vert.add_child(gui.Label(key))
        list_view = gui.ListView()
        list_view.set_max_visible_items(n)
        self.list_views[key] = list_view
        self.list_views[key].set_items(self.list_view_lists[key])
        vert.add_child(list_view)
        self.layout.add_child(vert)
        if callback:
            self.list_view_callbacks[key] = callback

    def add_to_list(self, key: str, values: List[str]):
        self.list_view_lists[key] += [i for i in values if i not in self.list_view_lists[key]]
        self.list_views[key].set_items(self.list_view_lists[key])
        if key in self.list_view_callbacks:
            self.list_view_callbacks[key]()

    def remove_from_list(self, key: str, values: List[str]):
        self.list_view_lists[key] = [i for i in self.list_view_lists[key] if i not in values]
        self.list_views[key].set_items(self.list_view_lists[key])
        if key in self.list_view_callbacks:
            self.list_view_callbacks[key]()

    def remove_selected_from_list(self, key):
        selected = self.list_views[key].selected_value
        self.remove_from_list(key, [selected])

    def clear_list(self, key):
        self.list_view_lists[key].clear()
        self.list_views[key].set_items(self.list_view_lists[key])
        if key in self.list_view_callbacks:
            self.list_view_callbacks[key]()

    def get_value(self, key: str) -> Union[str, float, int, Tuple[float, float, float]]:
        input_type = self.inputs_to_types[key]
        if input_type == str:
            return self.inputs[key].text_value
        elif input_type == float:
            return self.inputs[key].double_value
        elif input_type == int:
            return self.inputs[key].int_value
        elif input_type == gui.Button:
            return self.buttons[key].is_on
        elif input_type == gui.Checkbox:
            return self.checkboxes[key].checked
        elif input_type == gui.Color:
            rgb = tuple([
                    self.inputs[key].color_value.red,
                    self.inputs[key].color_value.green,
                    self.inputs[key].color_value.blue
            ])
            return rgb

    def set_value(self, key: str, value: Union[str, float, int, gui.Color, bool, None]):
        input_type = self.inputs_to_types[key]
        if type(value) != input_type:
            logger.debug(f"Attempt to set value with invalid type: {key}")
            value = self.DEFAULTS[input_type]
        if input_type == str:
            self.inputs[key].text_value = value
        elif input_type == float:
            self.inputs[key].double_value = value
        elif input_type == int:
            self.inputs[key].int_value = value
        elif input_type == gui.Color:
            self.inputs[key].color_value = value
        elif input_type == gui.Button:
            self.buttons[key].is_on = value
        elif input_type == gui.Checkbox:
            self.checkboxes[key].checked = value


class ButtonPool:
    def __init__(self, layout: gui.Vert):
        self.layout = layout
        self.buttons: Dict[Optional[str | int], gui.WidgetProxy] = dict()

    def add_button(self, key, button: gui.Button):
        button_layout = gui.Horiz()
        if key in self.buttons.keys():
            logger.debug(f"Replacing existing button")
            self.buttons[key].set_widget(button)
            return

        # check for hidden buttons to replace
        for existing_key, button_proxy in self.buttons.items():
            if not button_proxy.enabled:  # ok to replace
                logger.debug(f"Replacing {existing_key} with {key}")
                button_proxy.set_widget(button)
                self.buttons[key] = button_proxy
                del self.buttons[existing_key]
                return

        # finally if none to be replaced we add a new one
        logger.debug(f"Adding a new button for {key}")
        self.buttons[key] = gui.WidgetProxy()
        self.buttons[key].set_widget(button)
        button_layout.add_child(self.buttons[key])
        button_layout.add_stretch()
        self.layout.add_child(button_layout)

    def remove_button(self, key):
        if key in self.buttons:
            logger.debug(f"Disabling button {key}")
            self.buttons[key].enabled = False
            self.buttons[key].visible = False
        else:
            logger.debug(f"Button not found {key}")

    def clear(self):
        for name, button in self.buttons.items():
            button.enabled = False
            button.visible = False