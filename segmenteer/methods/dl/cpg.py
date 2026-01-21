from pathlib import Path
from segmenteer.core.base import PathSegmenter

class CPGSegmenter(PathSegmenter):

    def __init__(
		self,
		docker_image: str = "dodrio1.umcn.nl/daangeijs/tissueseg:latest",
		device: str | None = None,
        *args,
        **kwargs,
	):
        super().__init__(*args, **kwargs)
        self.docker_image = docker_image
        self.device = device
        self._validate_dependencies()
        import docker
        self._client = docker.from_env()

    def _validate_dependencies(self):
        """Check if docker and pyvips are available, raise helpful error if not."""
        try:
            import docker
        except ImportError:
            raise ImportError(
                "docker is required for CPGSegmenter but not installed. "
                "Install it with: pip install 'segmenteer[cpg]'"
            )
        try:
            import pyvips
        except ImportError:
            raise ImportError(
                "pyvips is required for CPGSegmenter but not installed. "
                "Install it with: pip install 'segmenteer[cpg]'"
            )


    @property
    def name(self) -> str:
        return "cpg_tissuemasker"

    def _segment_path(self, image_path: Path, output_path: Path) -> None:
        import docker
        
        command = [
            "-c",
            f"sh /home/user/run.sh /data/{image_path.name} /output/{output_path.name}",
        ]

        if self.device is not None:
            raise NotImplementedError("Only GPU support is implemented.")
        device_requests = [docker.types.DeviceRequest(device_ids=['0'], capabilities=[['gpu']])]

        # NOTE: Container is run from scratch, so has to load weights too every time.
        # NOTE: Sometimes it hangs if it cannot find the necessary 4.0 mpp (0.25 mpp tolerance) spacing.
        try:
            container = self._client.containers.run(
                self.docker_image,
                command,
                volumes={
                    str(image_path.parent.resolve()): {'bind': '/data', 'mode': 'rw'},
                    str(output_path.parent.resolve()): {'bind': '/output', 'mode': 'rw'},
                },
                remove=True,
                detach=True,
                entrypoint="/bin/bash",
                device_requests=device_requests,
                auto_remove=True,
            )
        except docker.errors.DockerException as e:
            raise RuntimeError(
                "Error running Docker container for CPGSegmenter. "
                "Make sure Docker is installed and running."
            ) from e
        for line in container.logs(stream=True):
            decoded = line.decode()
            print(decoded, end="")
            if "digitalpathology.errors.imageerrors.PixelSpacingLevelError" in decoded:
                raise RuntimeError(
                    "CPGSegmenter failed due to missing required MPP level. "
                    "Make sure the WSI contains a level close to 4.0 MPP."
                )