"""Run only the bundled fictional fixture. No connector or production inputs."""
from .fixtures import sample
from .publication import publish

if __name__ == "__main__":
    print(publish(sample()))
