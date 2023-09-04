subdir('common')

compiler_id = meson.get_compiler('cpp').get_id()

plugins_pch = custom_target('plugins-pch-pthread', input: 'pch/pch.h', output: 'pch-pthread.h.pch',
    #  command: [compiler_id, '-pthread', '-std=c++17', '-x', 'c++-header', '@INPUT@', '-o', '@OUTPUT@', '-Iplugins/single_plugins', '-I../plugins/single_plugins', '-I../src/api', '-I.', '-I..', '-Iplugins/common', '-I../plugins/common', '-Iplugins/vswitch', '-I../plugins/vswitch', '-Iplugins/wobbly', '-I../plugins/wobbly', '-Iplugins/grid', '-I../plugins/grid', '-Isubprojects/wf-touch', '-I../subprojects/wf-touch', '-I/usr/include/libdrm', '-I/usr/include/pixman-1', '-I/usr/include/libevdev-1.0', '-I/usr/include/libxml2', '-I/usr/include', '-I/usr/include/cairo', '-I/usr/include/libpng16', '-I/usr/include/freetype2', '-I/usr/include/harfbuzz', '-I/usr/include/glib-2.0', '-I/usr/lib64/glib-2.0/include', '-I/usr/include/sysprof-4', '-I/usr/include/pango-1.0', '-I/usr/include/libmount', '-I/usr/include/blkid', '-I/usr/include/fribidi', '-fdiagnostics-color=always', '-fno-omit-frame-pointer', '-D_FILE_OFFSET_BITS=64', '-Wall', '-Winvalid-pch', '-Wextra', '-std=c++17', '-O0', '-g', '-DWLR_USE_UNSTABLE', '-Wno-unused-parameter', '-DWF_USE_CONFIG_H', '-fPIC'])
command: [compiler_id, '-pthread', '-fPIC', '-std=c++17', '-x', 'c++-header', '@INPUT@', '-o', '@OUTPUT@'])
plugins_pch_dep = declare_dependency(sources: [plugins_pch])
plugins_pch_args = ['-include-pch', plugins_pch.full_path(), '-pthread']

subdir('ipc')
subdir('protocols')
subdir('vswitch')
subdir('wobbly')
subdir('grid')
subdir('decor')
subdir('animate')
subdir('cube')
subdir('window-rules')
subdir('blur')
subdir('tile')
subdir('wm-actions')
subdir('scale')
subdir('single_plugins')
