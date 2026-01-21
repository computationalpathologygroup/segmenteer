import os
from pathlib import Path
from segmenteer.core.utils import mask_to_geojson
from segmenteer.io.loader import Image
import tempfile
import geojson

class CPGSegmenter:

    def __init__(
		self,
		docker_image: str = "dodrio1.umcn.nl/daangeijs/tissueseg:latest",
		min_area: int = 10,
		device: str | None = None,
	):
        self.docker_image = docker_image
        self.min_area = min_area
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

    def segment(self, image: Image) -> geojson.FeatureCollection:
        import docker
        import pyvips
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_file = str(tmpdir / "tissuemask.tif")
            command = [
                "-c",
                f"sh /home/user/run.sh /data/{os.path.basename(image.path)} /output/{os.path.basename(output_file)}",
            ]

            if self.device is not None:
                raise NotImplementedError("Only GPU support is implemented.")
            device_requests = [docker.types.DeviceRequest(device_ids=['0'], capabilities=[['gpu']])]

            # NOTE: Container is run from scratch, so has to load weights too every time.
            # NOTE: Sometimes it hangs if it cannot find the necessary 4.0 mpp (0.25 mpp tolerance) spacing.
            container = self._client.containers.run(
                self.docker_image,
                command,
                volumes={
                    str(Path.cwd()): {'bind': '/data', 'mode': 'rw'},
                    str(tmpdir): {'bind': '/output', 'mode': 'rw'},
                },
                remove=True,
                detach=True,
                entrypoint="/bin/bash",
                device_requests=device_requests,
                auto_remove=True,
            )
            for line in container.logs(stream=True):
                print(line.decode(), end="")
            # TODO: is it fair that we need to read the output file here again?
            # It is similar to what we're doing with the non-containerized algorithms,
            # except we need to read from disk here.
            mask = pyvips.Image.new_from_file(output_file).numpy()
        return mask_to_geojson(mask, self.min_area)
