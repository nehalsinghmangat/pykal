from setuptools import setup

package_name = "pykal_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nehal Singh Mangat",
    maintainer_email="nehalsinghmangat.software@gmail.com",
    description="ROS2 wrapper around pykal",
    license="MIT",
    entry_points={
        "console_scripts": [
            "generate_signal_node = pykal_ros.generate_nodes.generate_signal_node:main"
        ],
    },
)
