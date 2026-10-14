# Custom warehouse bounding-box SDG

This example loads `IsaacWarehouse.usd`, finds the uniquely named
`RackLargeEmpty_A1` prim, labels it as `RackLargeEmpty_A1`, and captures it
from one randomized camera. It writes only RGB images, tight 2D bounding
boxes, and 3D bounding boxes.

Run it from an Isaac Sim installation with:

```bash
./python.sh standalone_examples/replicator/custom_workflow/custom_workflow.py \
    --config standalone_examples/replicator/custom_workflow/config.yaml
```

The USD path in `config.yaml` is machine-specific. Change `env_url` on the
remote machine if the warehouse is stored at a different location.
