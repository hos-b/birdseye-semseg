# image for the data collection program, because fuck my life

# base image
FROM ubuntu:20.04

# create user `mass`
RUN apt-get update
RUN apt-get install -y --no-install-recommends openssl sudo apt-utils
RUN useradd -rm -d /home/mass -s /bin/bash -g root -G sudo -u 1001 -p $(openssl passwd -1 mass) mass

# symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# non-interactive
ENV DEBIAN_FRONTEND noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# typical software
RUN apt-get install -y --no-install-recommends cmake git vim build-essential less 
RUN apt-get install -y --no-install-recommends pkg-config wget unzip tar openssh-client gnupg2 lsb-release

# installing ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get -y install ros-noetic-desktop-full
RUN apt-get -y install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN apt-get -y install python3-catkin-tools python3-osrf-pycommon
RUN rosdep init

# making directories
RUN mkdir -p /home/mass/catkin_ws/src/mass_data_collector
RUN mkdir -p /home/mass/libraries/libcarla-install
RUN mkdir -p /home/mass/data
RUN chown -R mass:root /home/mass

# setting up ROS
USER mass
RUN groups
WORKDIR /home/mass
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/mass/.bashrc
RUN /bin/bash -c "source /home/mass/.bashrc"
RUN rosdep update

# building HDF5 12.0 library
WORKDIR /home/mass/libraries
RUN wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_0/source/hdf5-1.12.0.tar.gz --no-check-certificate
RUN tar -xvf hdf5-1.12.0.tar.gz
RUN mkdir -p /home/mass/libraries/hdf5-1.12.0/build
WORKDIR /home/mass/libraries/hdf5-1.12.0/build
RUN cmake ..
RUN make -j8
USER root
RUN make install
USER mass

# building nanoflann
WORKDIR /home/mass/libraries
RUN wget https://github.com/jlblancoc/nanoflann/archive/v1.3.2.tar.gz --no-check-certificate
RUN tar -xvf v1.3.2.tar.gz
RUN mkdir -p /home/mass/libraries/nanoflann-1.3.2/build
WORKDIR /home/mass/libraries/nanoflann-1.3.2/build
RUN cmake ..
RUN make -j8
USER root
RUN make install
USER mass

# library paths
RUN echo "export LIBHDF5_ROOT=/usr/local/HDF_Group/HDF5/1.12.0" >> /home/mass/.bashrc
RUN echo "export CARLA_910_CXX_API=/home/mass/libraries/libcarla-install" >> /home/mass/.bashrc

# finalizing
WORKDIR /home/mass
RUN echo "Done! use: docker run -v $CARLA_910_CXX_API:/home/mass/libraries/libcarla-install:ro -v /export/ad-lab/repos/birdseye-semseg/mass_data_collector:/home/mass/catkin_ws/src/mass_data_collector:rw -v ~/mass-data:/home/mass/data:rw --net=host -it noetic-mass"
