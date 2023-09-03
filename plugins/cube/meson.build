animiate = shared_module('cube',
                         ['cube.cpp', 'cubemap.cpp', 'skydome.cpp', 'simple-background.cpp'],
                         include_directories: [wayfire_api_inc, wayfire_conf_inc, plugins_common_inc],
                         dependencies: [wlroots, pixman, wfconfig, plugins_pch_dep],
                         cpp_args: plugins_pch_args,
                         install: true,
                         install_dir: join_paths(get_option('libdir'), 'wayfire'))
