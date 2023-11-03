import os
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "1.0.2"

SETUP_DIRECTORY = Path(__file__).resolve().parent
class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    EIGEN3_DIRNAME = Path("./boms/eigen-3.4.0")

    def __str__(self) -> str:
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)

        if eigen_include_dir is not None:
            return eigen_include_dir

        target_dir = SETUP_DIRECTORY.joinpath(self.EIGEN3_DIRNAME) #SETUP_DIRECTORY / self.EIGEN3_DIRNAME
        if target_dir.exists():
            return str(target_dir) #target_dir.name

        download_target_dir = SETUP_DIRECTORY / "eigen3.zip"
        import zipfile

        import requests

        response = requests.get(self.EIGEN3_URL, stream=True)
        with download_target_dir.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)

        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall(target_dir.parent)

        os.remove(download_target_dir)

        return str(target_dir) #target_dir.name

cpp_args = ['-std=c++17', '-mavx512f', '-mfma', '-fopenmp', '-O3', '-lpthread']

ext_modules = [
    Pybind11Extension("boms_wrapper",
        ["./boms/boms_wrapper.cpp", "./boms/meanshift.cpp"],
        extra_compile_args = cpp_args,
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs=[get_eigen_include()],
        depends=["./boms/meanshift.hpp"],
        ),
]

setup(
    name='boms',
    version=__version__,
    author='Ocima Kamboj',
    author_email='ocimakamboj@gmail.com',
    description='Cell Segmentation for Spatial Transcriptomics Data using BOMS',
    packages=['boms'],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires='>=3.9',
    install_requires = [
        "numpy",
        "mkl",
        "mkl-service",
        "scipy",
        ]
)