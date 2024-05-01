# AeroGym

AeroGym is a Python package that provides a set of Farama-Foundation/Gymnasium environments to apply reinforcement learning to unsteady aerodynamics problems. Currently, two environments are available for a flat plate pitching problem:
1. A Wagner environment
2. A viscous flow environment. This environment relies on [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/) to interface with the [ViscousFlow](https://github.com/JuliaIBPM/ViscousFlow.jl) package in Julia.

## Installation

### Prerequisites

Before installing AeroGym, ensure you have Python (version >= 3.7) and Julia (version >=1.6.1) installed. 

Your Julia environment should have the [ViscousFlow](https://github.com/JuliaIBPM/ViscousFlow.jl) package installed (version >= 0.6.6).

### Installing AeroGym

AeroGym can be installed from source using `pip install`. Here's how you can install it:
1. **Clone this repository to your local machine:**
   ```bash
   git clone https://github.com/diederikb/AeroGym.git
   ```
2. **Navigate to the directory containing the cloned repository:**
   ```bash
   cd AeroGym
   ```
3. **Install AeroGym using pip, preferably in a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/):**
   ```bash
   pip install .
   ```
   You can also install AeroGym in editable mode (`pip install -e .`), allowing you to make changes to the source code if needed without needing to re-install. See the [pip documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for further installation details.

## Usage
To use AeroGym in your Python projects, you can import it like any other Python package:

```python
import AeroGym
```
Please refer to the [Jupyter notebooks](aero_gym/notebooks) for detailed usage instructions and examples. If you want to run the notebooks, you will have to install [Jupyter](https://jupyter.org/install) and install a kernel for your virtual environment if you're using one (by running `ipython kernel install --user --name=NAME_OF_YOUR_ENVIRONMENT` after activating your environment). Make sure to change the notebook kernel to your 
