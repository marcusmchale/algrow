import logging
from .custom_types import arg_types


class TemporaryArgsAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['temporary_args'], msg), kwargs


logger = logging.getLogger(__name__)


def update_arg(args, arg, val, temporary=False):
    # We need to distinguish updates to a temporary copy of args vs the main args
    if temporary:
        adapter = TemporaryArgsAdapter(logger, {'temporary_args': "Temporary"})
        # only want to log this at debug level
    else:
        adapter = logger

    if isinstance(val, list) and isinstance(val[0], tuple):
        val_str = f'{[",".join([str(j) for j in i]) for i in val]}'.replace("'", '"')
        # coerce to known type:
        try:
            val = [arg_types[arg](v) for v in val]
        except ValueError:
            adapter.debug("Issue with updating arg")
            raise
    else:
        if isinstance(val, tuple):
            val_str = f"\"{','.join([str(i) for i in val])}\""
        else:
            val_str = str(val)
            # coerce to known type:
        try:
            val = arg_types[arg](val)
        except ValueError:
            adapter.debug("Issue with updating arg")
            raise

    if vars(args)[arg] is None:
        if temporary:
            adapter.debug(f"Setting {arg}: {val_str}")
        else:
            adapter.info(f"Setting {arg}: {val_str}")
        vars(args).update({arg: val})
        adapter.debug(f"{arg}:{vars(args)[arg]}")
    else:
        if vars(args)[arg] == val:
            adapter.debug(f"Existing value matches the update so no change will be made {arg}: {val}")
        else:
            if temporary:
                adapter.debug(f"Overwriting configured value for {arg}: {vars(args)[arg]} will be set to {val}")
            else:
                adapter.info(f"Overwriting configured value for {arg}: {vars(args)[arg]} will be set to {val}")
            vars(args).update({arg: val})


def minimum_calibration(args):
    return all([
        (args.hull_vertices is not None and len(args.hull_vertices) >= 4),
        args.scale is not None
    ])


def layout_defined(args):
    return all([
        args.circle_colour is not None,
        args.circle_diameter is not None,
        args.plate_circle_separation is not None,
        args.plate_width is not None,
        args.circles_per_plate is not None,
        args.n_plates is not None,
        args.circles_cols_first is not None,
        args.circles_right_left is not None,
        args.circles_bottom_top is not None,
        args.plates_cols_first is not None,
        args.plates_bottom_top is not None,
        args.plates_right_left is not None
    ])


def calibration_complete(args):
    if args.whole_image:
        return minimum_calibration(args)
    else:
        return all([
            args.scale is not None,
            (args.hull_vertices is not None and len(args.hull_vertices) >= 4),
            layout_defined(args)
        ])
