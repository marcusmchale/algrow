area_header = ['filename', 'plate', 'well', 'pixels', 'mm²']
datetime_regex = r".*(?P<year>[0-9]{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12][0-9]|3[01])_(?P<hour>[01][0-9]|2[0-4])h(?P<minute>[0-5][0-9])m.*"  # year, month, day, hour, minute
tank_regex = r".*pi([0-9]).*"
