# Docker
## 1. Installation
There are two methods of installing Docker, and either of them should work.
The first is the standard way when you have root access on your machine (i.e., you can run `sudo`).
The second is without root access, which is slightly more limiting (although we have not encountered any of those limitations while developing Repro).
All of this may look scary, but it's really not.

### 1.1 With Root Access
Installing with root access is pretty straightforward, and we refer you to [the official Docker instructions](https://docs.docker.com/get-docker/).

### 1.2 Without Root Access
Install without root access can be a trickier, but it is doable.
The official Docker instructions are located [here](https://docs.docker.com/engine/security/rootless/).
We are currently developing Repro in rootless mode on openSUSE and have encountered no blocking issues.
Please see the [troubleshooting](#troubleshooting) section below if you run into any issues.

### 1.3 Verifying the Installation
First, start the Docker daemon, which needs to be running whenever you run Repro:
```shell script
# with root access
dockerd

# without root access
systemctl --user start docker.service
```

Then, run the following shell script.
It will create a Docker image which is just a copy of the [the official Python 3.7 image](https://hub.docker.com/_/python) image.
It will then run a container and open up an interactive bash shell so that you can run commands 
```shell script
# Create a new directory
mkdir docker-test
cd docker-test

# Create the Dockerfile
cat > Dockerfile << EOL
FROM python:3.7
EOL

# Build the image
docker build -t docker-test .

# Run the image with an interactive shell
docker run -it docker-test /bin/bash
```

Then, within the shell, you can run
```shell script
python --version
```
to verify that Python 3.7 is running in the container.
If this worked correctly, your Docker installation should be correct.
Run `exit` to exit the container's shell.

### 1.4 GPU Access
In order to enable your containers to access the host machine's GPUs, the [Nivida Container Toolkit](https://github.com/NVIDIA/nvidia-docker) ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)) needs to be installed.
This requires `sudo` access, so if you do not have root access, you need to ask your system administrator to install it.

Once it is installed, you need to edit the daemon configuration file.
The default location for this is `/etc/docker/daemon.json` (we have not tested this) if you installed with root access, otherwise it is `~/.config/docker/daemon.json`.
Add the following entry to the file (creating it, if necessary):
```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```
You need to restart the daemon once the file has been changed.

#### 1.4.1 Verifying the GPU Installation
After installing the Nvidia Container Toolkit, you can verify your containers can access the host's GPUs by running the following script.
It will create a Docker image that has the Nvidia drivers configured, install PyTorch, then open up an interactive shell:
```shell script
# Create a new directory
mkdir gpu-test
cd gpu-test

# Create the Dockerfile
cat > Dockerfile << EOL
FROM pure/python:3.7-cuda10.0-base
RUN pip install --no-cache-dir torch==1.9.0
EOL

# Build the image
docker build -t gpu-test .

# Run the image with an interactive shell, setting the runtime to "nvidia" to allow GPU access
docker run -it --runtime nvidia gpu-test /bin/bash
```

A commandline prompt should now appear.
Within the Docker container, run:
```shell script
# Verify that "nvidia-smi" works
nvidia-smi

# Verify torch has GPU access
python -c "import torch; print(torch.cuda.is_available())"
```
The key difference between this test and the general Docker install verification is the `--runtime nvidia` that is passed to the `docker run` command.
This is necessary if you want the container to be able to access the GPUs.
If you do not include this flag, the container will still run, but it won't have GPU access.

### 1.5 Repro-Specific Configuration
We believe the following configuration only applies to users who installed Docker without root access, although we have not tested it on a setup with root acces.

Repro needs to know the url of the Docker server which is running.
This is done via configuration file: `~/.repro/config.json`.
You can override the location of this file by setting the `REPRO_CONFIG` environment variable to whatever file you want.

Rootless installation users should update the `config.json` with this entry:
```json
{
  "docker_server": "unix:///run/user/50868/docker.sock"
}
```
where the value is equal to the `DOCKER_HOST` environment variable that you set during the rootless installation.

### 1.6 [Troubleshooting](#troubleshooting)
- If you fail the rootless installation "prerequisites" section about `/etc/subuid` and `/etc/subgid`, then ask your system admin to edit those files to add you.
Our system admin did not have any issues with editing them for us.

- If you encounter an error that says `lchown /etc/gshadow: operation not permitted`, then you may need to edit the location where Docker stores the images it builds.
It is required to be a location not on an NFS file system (i.e., the disk needs to be local to the machine).
In order to do that, edit the `daemon.json` to include the following entry:
  ```json
  {
    "data-root": "/path/on/not-nfs"
  }
  ```
  Restart the daemon and try again.
  This solution was based on [this issue](https://github.com/docker/for-linux/issues/1172).

## 2. Docker Command Cheat Sheet
Repro wraps any interaction with Docker that a user needs to do.
However, if you need to interact directly with Docker, here are some useful commands:
- Start and stop the Docker daemon
  ```shell script
  # With root access (ctrl-c to stop)
  dockerd
  
  # Without root access
  systemctl --user <start,stop> docker.service
  ```

- Build an image (must be run in the directory where the `Dockerfile` is):
  ```shell script
  docker build -t <name> .
  ```
  
- Run a container with an interactive shell:
  ```shell script
  docker run -it <name> /bin/bash
  
  # With GPU access
  docker run -it --runtime nvidia /bin/bash
  ```
  
- See the list of images:
  ```shell script
  docker image ls
  ```
  
- Delete an image:
  ```shell script
  docker image rm [--force] <name>
  ```
