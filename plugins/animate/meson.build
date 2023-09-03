animiate = shared_module('animate',
                         ['animate.cpp',
                          'fire/particle.cpp',
                          'fire/fire.cpp'],
                         include_directories: [wayfire_api_inc, wayfire_conf_inc],
                         cpp_args: plugins_pch_args,
                         dependencies: [wlroots, pixman, wfconfig, plugins_pch_dep],
                         install: true,
                         install_dir: join_paths(get_option('libdir'), 'wayfire'))
