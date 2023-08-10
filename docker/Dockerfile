FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
COPY install_ros2.sh /
RUN \
  export DEBIAN_FRONTEND=noninteractive; \
  apt update; \
  apt upgrade -y; \
  apt install python3 python3-pip jupyter tmux sqlite3 -y; \
  python3 -m pip install scipy torch tqdm ray[tune]; \
  sh /install_ros2.sh; \
  rm /install_ros2.sh; \
  apt autoremove -y; \
  apt clean -y
