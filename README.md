# MiDas Depth Solver
A Docker container for determining depth maps of 2D images using [MiDaS](https://github.com/isl-org/MiDaS).

## Usage

```sh
nvidia-docker run -it -v $(pwd)/test_img:/input -v $(pwd)/output:/output -v midas_cache:/root/.cache ewpratten/midas-depth-solver
```

The following environment variables are supported:

| Variable      | Description                                                                                        |
|---------------|----------------------------------------------------------------------------------------------------|
| `MIDAS_MODEL` | Model size. Can be one of: **DPT_Large**, **DPT_Hybrid**, or **MiDaS_small** (default `DPT_Large`) |
| `FORCE_CPU_COMPUTE` | If set, forces CPU computation (default unset) |

## Examples

Input:

![Input](test_img/test.jpg)

Output:

![Output](output/test.jpg)
