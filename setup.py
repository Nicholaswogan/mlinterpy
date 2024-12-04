from skbuild import setup

with open("CMakeLists.txt",'r') as fil:
    lines = fil.readlines()
    for line in lines:
        if line.startswith("project(MLINTERPY"):
            version = line.split('"')[1]
            break
            
setup(
    name="mlinterpy",
    packages=['mlinterpy'],
    python_requires='>=3.6',
    version=version,
    license="GNU General Public License v3.0",
    install_requires=['numpy'], 
    author='Nicholas Wogan',
    author_email = 'nicholaswogan@gmail.com',
    description = "Multidimensional linear interpolation",
)


