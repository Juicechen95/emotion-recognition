# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/orange/Expression_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/orange/Expression_demo

# Include any dependencies generated for this target.
include CMakeFiles/fer_train_CK.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fer_train_CK.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fer_train_CK.dir/flags.make

CMakeFiles/fer_train_CK.dir/train_CK.cpp.o: CMakeFiles/fer_train_CK.dir/flags.make
CMakeFiles/fer_train_CK.dir/train_CK.cpp.o: train_CK.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/orange/Expression_demo/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/fer_train_CK.dir/train_CK.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/fer_train_CK.dir/train_CK.cpp.o -c /home/orange/Expression_demo/train_CK.cpp

CMakeFiles/fer_train_CK.dir/train_CK.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fer_train_CK.dir/train_CK.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/orange/Expression_demo/train_CK.cpp > CMakeFiles/fer_train_CK.dir/train_CK.cpp.i

CMakeFiles/fer_train_CK.dir/train_CK.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fer_train_CK.dir/train_CK.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/orange/Expression_demo/train_CK.cpp -o CMakeFiles/fer_train_CK.dir/train_CK.cpp.s

CMakeFiles/fer_train_CK.dir/train_CK.cpp.o.requires:
.PHONY : CMakeFiles/fer_train_CK.dir/train_CK.cpp.o.requires

CMakeFiles/fer_train_CK.dir/train_CK.cpp.o.provides: CMakeFiles/fer_train_CK.dir/train_CK.cpp.o.requires
	$(MAKE) -f CMakeFiles/fer_train_CK.dir/build.make CMakeFiles/fer_train_CK.dir/train_CK.cpp.o.provides.build
.PHONY : CMakeFiles/fer_train_CK.dir/train_CK.cpp.o.provides

CMakeFiles/fer_train_CK.dir/train_CK.cpp.o.provides.build: CMakeFiles/fer_train_CK.dir/train_CK.cpp.o

CMakeFiles/fer_train_CK.dir/fer.cpp.o: CMakeFiles/fer_train_CK.dir/flags.make
CMakeFiles/fer_train_CK.dir/fer.cpp.o: fer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/orange/Expression_demo/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/fer_train_CK.dir/fer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/fer_train_CK.dir/fer.cpp.o -c /home/orange/Expression_demo/fer.cpp

CMakeFiles/fer_train_CK.dir/fer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fer_train_CK.dir/fer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/orange/Expression_demo/fer.cpp > CMakeFiles/fer_train_CK.dir/fer.cpp.i

CMakeFiles/fer_train_CK.dir/fer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fer_train_CK.dir/fer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/orange/Expression_demo/fer.cpp -o CMakeFiles/fer_train_CK.dir/fer.cpp.s

CMakeFiles/fer_train_CK.dir/fer.cpp.o.requires:
.PHONY : CMakeFiles/fer_train_CK.dir/fer.cpp.o.requires

CMakeFiles/fer_train_CK.dir/fer.cpp.o.provides: CMakeFiles/fer_train_CK.dir/fer.cpp.o.requires
	$(MAKE) -f CMakeFiles/fer_train_CK.dir/build.make CMakeFiles/fer_train_CK.dir/fer.cpp.o.provides.build
.PHONY : CMakeFiles/fer_train_CK.dir/fer.cpp.o.provides

CMakeFiles/fer_train_CK.dir/fer.cpp.o.provides.build: CMakeFiles/fer_train_CK.dir/fer.cpp.o

# Object files for target fer_train_CK
fer_train_CK_OBJECTS = \
"CMakeFiles/fer_train_CK.dir/train_CK.cpp.o" \
"CMakeFiles/fer_train_CK.dir/fer.cpp.o"

# External object files for target fer_train_CK
fer_train_CK_EXTERNAL_OBJECTS =

fer_train_CK: CMakeFiles/fer_train_CK.dir/train_CK.cpp.o
fer_train_CK: CMakeFiles/fer_train_CK.dir/fer.cpp.o
fer_train_CK: CMakeFiles/fer_train_CK.dir/build.make
fer_train_CK: /usr/local/lib/libopencv_calib3d.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_core.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_features2d.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_flann.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_highgui.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_imgproc.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_ml.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_objdetect.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_photo.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_shape.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_stitching.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_superres.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_video.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_videoio.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_videostab.so.3.2.0
fer_train_CK: /usr/local/lib/x86_64-linux-gnu/libdlib.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libnsl.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libSM.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libICE.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libX11.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libXext.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libpng.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libjpeg.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libnsl.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libSM.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libICE.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libX11.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libXext.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libpng.so
fer_train_CK: /usr/lib/x86_64-linux-gnu/libjpeg.so
fer_train_CK: /usr/local/lib/libopencv_objdetect.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_calib3d.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_features2d.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_flann.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_highgui.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_ml.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_photo.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_video.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_videoio.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_imgproc.so.3.2.0
fer_train_CK: /usr/local/lib/libopencv_core.so.3.2.0
fer_train_CK: CMakeFiles/fer_train_CK.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable fer_train_CK"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fer_train_CK.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fer_train_CK.dir/build: fer_train_CK
.PHONY : CMakeFiles/fer_train_CK.dir/build

CMakeFiles/fer_train_CK.dir/requires: CMakeFiles/fer_train_CK.dir/train_CK.cpp.o.requires
CMakeFiles/fer_train_CK.dir/requires: CMakeFiles/fer_train_CK.dir/fer.cpp.o.requires
.PHONY : CMakeFiles/fer_train_CK.dir/requires

CMakeFiles/fer_train_CK.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fer_train_CK.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fer_train_CK.dir/clean

CMakeFiles/fer_train_CK.dir/depend:
	cd /home/orange/Expression_demo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/orange/Expression_demo /home/orange/Expression_demo /home/orange/Expression_demo /home/orange/Expression_demo /home/orange/Expression_demo/CMakeFiles/fer_train_CK.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fer_train_CK.dir/depend

