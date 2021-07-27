## Installation
There are two methods of installing Docker, and either of them should work.
The first is the standard way when you have root access on your machine (i.e., you can run `sudo`).
The second is without root access, which is slightly more limiting (although we have not encountered any of those limitations while developing Repro).

### With Root Access
Installing with root access is pretty straightforward, and we refer you to [the official Docker instructions](https://docs.docker.com/get-docker/).

### Without Root Access
Install without root access can be a trickier, but it is doable.
The official Docker instructions are located [here](https://docs.docker.com/engine/security/rootless/).
We are currently developing Repro in rootless mode on openSUSE and have encountered no blocking issues.
Please see the [troubleshooting](#troubleshooting) section below if you run into any issues.

### Verifying the Installation
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


### Repro-Specific Configuration
`~/.repro/config.json`. `REPRO_CONFIG` environment variable
```json
{
	"docker_server": "unix:///run/user/50868/docker.sock"
}
```

### GPU Access
Nvidia container runtime only supported Linux.
You should still be able to build the Dockerfiles (unconfirmed) and run them on other operating systems, but you will not be able to get GPU access.


`~/.config/docker/daemon.json`, stop and start daemon
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

#### Verifying the GPU Installation
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

### Repro-Specific Configuration
`~/.repro/config.json`. `REPRO_CONFIG` environment variable
```json
{
	"docker_server": "unix:///run/user/50868/docker.sock"
}
```

### [Troubleshooting](#troubleshooting)
NFS: `lchown /etc/gshadow: operation not permitted`
`~/.config/docker/daemon.json`, stop and start daemon
```json
{
	"data-root": "/path/on/not-nfs"
}
```

If you fail the "prerequisites" section about `/etc/subuid` and `/etc/subgid`, then ask your system admin to edit those files to add you.


## Docker Command Cheat Sheet
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