#!/bin/bash

xhost +

XAUTH=/tmp/.docker.xauth
# export XAUTH=$XAUTH

echo "Preparing Xauthority data..."
xauth_list=$(xauth nlist :0 | tail -n 1 | sed -e 's/^..../ffff/')
if [ ! -f $XAUTH ]; then
    if [ ! -z "$xauth_list" ]; then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

echo "Done."
echo ""
echo "Verifying file contents:"
file $XAUTH
echo "--> It should say \"X11 Xauthority data\"."
echo ""
echo "Permissions:"
ls -FAlh $XAUTH
echo ""
echo "Running docker..."

docker run \
    -it --rm \
    --privileged \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --gpus="all" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$PWD/../:/app/ros2_ws/src/:rw" \
    --volume="/home/jack/.keras/:/app/.keras/:rw" \
    --volume="/dev:/dev:rw" \
    --name="dreamer" \
    --network="bridge" \
    -p 2002:2002 \
    -p 3003:3003 \
    -p 4004:4004 \
    -p 5005:5005 \
    -p 6006:6006 \
    -p 7007:7007 \
    --memory-swap="-1" \
    --runtime=nvidia \
    dreamer:latest


