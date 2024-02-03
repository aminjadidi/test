"""Canonical neural network layers."""
import copy
import enum
import functools
import math
import pprint

from absl import logging

import google3.base.python.pywrapbase  # pylint: disable=unused-import
from google3.platforms.deepsea.ffds.art.core.types import common as common_types
from google3.platforms.deepsea.ffds.art.core.types import roofline_cost
from google3.platforms.deepsea.ffds.art.core.types import sharding as sharding_types
from google3.platforms.deepsea.ffds.art.core.types import tensor as tensor_types
from google3.platforms.deepsea.ffds.art.frame import collectives

DS = tensor_types.dimension_strings_to_tensors_dict

# TODO(suvinay): Generalize the logging template to a logger class pattern, so
# we don't keep typing self.logprefix and other redundant text.
logging.use_cpp_logging()


def leaf_operators(
    operator: 'Operator', leaves: list['Operator'] | None = None
) -> list['Operator']:
  """Get leaf operators of an operator (tree)."""
  if leaves is None: leaves = []
  if operator.operators is None:
    leaves.append(operator)
  else:
    for l in operator.operators:
      leaves = leaf_operators(l, leaves)
  return leaves


class Operator:
  """Defines generic operator."""

  # TODO(suvinay): Update documentation.
  # The expectation is that input_*, output_*
  # are specified with sharding (if applicable).
  # The operator is responsible for determining the collected sizes.
  def __init__(
      self,
      otype: str | None = 'Null',
      name: str | None = 'Null',
      inputs: dict[str, tensor_types.TensorAttributes] | None = None,
      outputs: dict[str, tensor_types.TensorAttributes] | None = None,
      pipeline_config: sharding_types.PipelineConfig = sharding_types.PipelineConfig.NONE,
      hw_multipliers: dict[str, float] | None = None,
      moniker: str | None = None,
      repeat: int = 1,
  ):
    self.type = otype
    self.name = name
    self.inputs = inputs or {}
    self.outputs = outputs or {}
    self.collected_inputs = None
    self.collected_outputs = None
    # Whether to pipeline the output of this operator to the next layer and over
    # which type of network (ICI or DCN)
    self.pipeline_config = pipeline_config
    self.hw_multipliers = hw_multipliers or {}
    # TODO(suvinay): Generalize monikers to allow multiple categories depending
    # on the analysis type.
    self.moniker = moniker
    self.operators = None
    self.repeat = repeat
    # TODO(suvinay): Remove sparsity once we infer this from
    # input tensors' sparsity in hw_multipliers.
    self.sparsity = None

  # TODO(suvinay): Update to return dict[str, float] as:
  # {'mxu': xxx, 'vpu': yyy}
  def compute_flops(self) -> float:
    return 0.0

  def compute_bytes(self) -> dict[str, float]:
    return {}

  def memory_bytes(self) -> roofline_cost.MemoryBytes:
    return roofline_cost.MemoryBytes()

  def peak_memory_bytes(self) -> roofline_cost.MemoryBytes:
    return roofline_cost.MemoryBytes()

  def communication_bytes(
      self,
  ) -> dict[str, tuple[float, tensor_types.Dimension]]:
    return {}

  def pipeline_bytes_per_device(self) -> float:
    return 0.0


class Einsum(Operator):
  """Generic sharded Einsum operator."""
  logprefix = '[art] [einsum]'

  def __init__(
      self,
      name: str,
      inputs: dict[str, tensor_types.TensorAttributes],
      outputs: dict[str, tensor_types.TensorAttributes],
      hw_multipliers: dict[str, float] | None = None,
      pipeline_config: sharding_types.PipelineConfig = sharding_types.PipelineConfig.NONE,
      moniker: str | None = None,
      repeat: int = 1,
  ):
    super().__init__(
        'Einsum', name, inputs, outputs, pipeline_config,
        hw_multipliers, moniker, repeat
    )
    logging.vlog(2, '%s Creating sharded einsum for %s',
                 self.logprefix, name)
    logging.vlog(2, '%s \tInputs: %s',
                 self.logprefix, pprint.pformat(self.inputs))
    logging.vlog(2, '%s \tOutputs: %s',
                 self.logprefix, pprint.pformat(self.outputs))
    self.initialize_collectives()
    logging.vlog(2, '%s \tCollected inputs: %s',
                 self.logprefix, pprint.pformat(self.collected_inputs))
    logging.vlog(2, '%s \tCollected outputs: %s',
                 self.logprefix, pprint.pformat(self.collected_outputs))
    self._parallel_layers = 1
    # TODO(suvinay): Document that this assumes 'p' dimensions are never
    # collected. Annotate in initialize_collectives() as well.
    for count, (_, v) in enumerate(self.inputs.items()):
      if count == 0 and v.ttype != common_types.TensorType.WEIGHTS:
        continue
      for dim in v.dimensions:
        # TODO(suvinay): Check if this holds when parallel layers are sharded.
        # Currently the common case is parallel layers dimension is not sharded
        # so the below should work.
        if dim.attr is tensor_types.DimensionAttribute.P:
          self._parallel_layers *= dim.size

  @property
  def ff_parallel_layers(self) -> int:
    """Number of parallel layers in feedforward layers."""
    return self._parallel_layers

  def compute_flops(self) -> float:
    input_tensors = (
        self.inputs if self.collected_inputs is None else self.collected_inputs
    )
    output_tensors = (
        self.outputs
        if self.collected_outputs is None
        else self.collected_outputs
    )
    red_dims = []
    # Pick any of the two input tensors to get the reduction dimensions.
    # At initialization we have already sanity checked the reduction dimensions
    # match for all (both) inputs.
    ip_tensor = next(iter(input_tensors.values()))
    for dim in ip_tensor.dimensions:
      if dim.attr is tensor_types.DimensionAttribute.R: red_dims.append(dim)

    # Pick any (the only) output tensor's dimensions to get the number of output
    # elements.
    output_size_and_ways = tensor_types.aggregate_size_and_ways(
        next(iter(output_tensors.values())).dimensions
    )

    reduction_size_and_ways = tensor_types.aggregate_size_and_ways(red_dims)
    output_elements = output_size_and_ways.size / output_size_and_ways.ways
    reduction_elements = (
        reduction_size_and_ways.size / reduction_size_and_ways.ways
    )
    flops = (output_elements *
             reduction_elements * self._parallel_layers * 2)
    logging.vlog(
        2,
        '%s %s compute flops: %.2f  [out-elements: %d, red-elements: %d]',
        self.logprefix, self.name,
        flops, output_elements, reduction_size_and_ways.size)
    return flops

  def compute_bytes(self) -> dict[str, float]:
    total_bytes = {
        'hbm': 0.0,
        'emem': 0.0,
        'vmem_read': 0.0,
        'vmem_write': 0.0,
        'wmem_read': 0.0,
    }
    input_tensors = (
        self.inputs if self.collected_inputs is None else self.collected_inputs
    )
    output_tensors = (
        self.outputs
        if self.collected_outputs is None
        else self.collected_outputs
    )
    # If the tensor is stored in a memory hierarchy further from VMEM or WMEM,
    # it will have to be staged in HBM.
    for _, v in input_tensors.items():
      if v.storage_location == common_types.MemoryType.VMEM:
        total_bytes['vmem_read'] += v.bytes()
      elif v.storage_location == common_types.MemoryType.WMEM:
        total_bytes['wmem_read'] += v.bytes()
      else:
        total_bytes['hbm'] += v.bytes()
    for _, v in output_tensors.items():
      assert (
          v.storage_location != common_types.MemoryType.WMEM
      ), 'No writes allowed to WMEM storage location'
      if v.storage_location == common_types.MemoryType.VMEM:
        total_bytes['vmem_write'] += v.bytes() * self._parallel_layers
      else:
        total_bytes['hbm'] += v.bytes() * self._parallel_layers
    logging.vlog(2, '%s \tcompute_bytes: %s', self.logprefix, total_bytes)
    # External memory offloading:
    for _, v in self.inputs.items():
      if (
          v.storage_location == common_types.MemoryType.EMEM
          and v.ttype == common_types.TensorType.WEIGHTS
      ):
        total_bytes['emem'] += v.bytes()

    for _, v in self.outputs.items():
      if (
          v.storage_location == common_types.MemoryType.EMEM
          and v.ttype == common_types.TensorType.ACTIVATIONS
      ):
        total_bytes['emem'] += v.bytes() * self._parallel_layers

    return total_bytes

  def _calc_memory_bytes(
      self,
      in_tensors: dict[str, tensor_types.TensorAttributes],
      out_tensors: dict[str, tensor_types.TensorAttributes],
      parallel_layers: int,
  ) -> roofline_cost.MemoryBytes:
    """Compute the number of bytes of input and output tensors, grouped.

    Group by tensor type (activation, weight, gradient) and memory location
    (hbm, vmem, emem).

    Args:
      in_tensors: input tensors to compute memory bytes for.
      out_tensors: output tensors to compute memory bytes for.
      parallel_layers: number of parallel layers.

    Returns:
      A MemoryBytes object grouping memory bytes by tensor type and location.
    """
    memory_bytes = roofline_cost.MemoryBytes()
    for _, v in in_tensors.items():
      memory_bytes.accumulate(v.ttype, v.storage_location, v.bytes())
    for _, v in out_tensors.items():
      if v.ttype == common_types.TensorType.ACTIVATIONS:
        memory_bytes.accumulate(
            v.ttype, v.storage_location, v.bytes() * parallel_layers
        )
      else:
        memory_bytes.accumulate(v.ttype, v.storage_location, 1)
    return memory_bytes

  def memory_bytes(
      self,
      count_input_activations: bool = True,
      count_output_activations: bool = True,
  ) -> roofline_cost.MemoryBytes:
    """Compute resident total memory footprint of inputs/outputs.

    Group by tensor type (activation, weight, gradient) and memory location
    (hbm, vmem, emem).

    Args:
      count_input_activations: boolean determining whether input activations
        should be accounted for.
      count_output_activations: boolean determining whether output activations
        should be accounted for.

    Returns:
      A MemoryBytes object grouping memory bytes by tensor type and location.
    """
    input_tensors = self.inputs
    if not count_input_activations:
      input_tensors = {
          k: v
          for k, v in self.inputs.items()
          if v.ttype == common_types.TensorType.WEIGHTS
      }
    output_tensors = self.outputs if count_output_activations else {}
    return self._calc_memory_bytes(
        input_tensors, output_tensors, self._parallel_layers
    )

  def peak_memory_bytes(
      self,
      count_input_activations: bool = True,
      count_output_activations: bool = True,
  ) -> roofline_cost.MemoryBytes:
    """Compute peak total memory footprint of inputs/outputs.

    Group by tensor type (activation, weight, gradient) and memory location
    (hbm, vmem, emem).

    Args:
      count_input_activations: boolean determining whether input activations
        should be accounted for.
      count_output_activations: boolean determining whether output activations
        should be accounted for.

    Returns:
      A MemoryBytes object grouping memory bytes by tensor type and location.
    """

    input_tensors = self._calc_emem_peak_memory_bytes_staged_in_hbm()

    if not count_input_activations:
      input_tensors = {
          k: v
          for k, v in input_tensors.items()
          if v.ttype == common_types.TensorType.WEIGHTS
      }
    output_tensors = {}
    if count_output_activations:
      output_tensors = self._calc_emem_peak_memory_bytes_staged_in_hbm(
          inputs=False
      )
    return self._calc_memory_bytes(
        input_tensors, output_tensors, self._parallel_layers
    )

  def _calc_emem_peak_memory_bytes_staged_in_hbm(
      self,
      inputs: bool = True,
  ) -> dict[str, tensor_types.TensorAttributes]:
    """Stages the input/ output emem tensors in hbm.

    Args:
      inputs: boolean to select if we want to process inputs or outputs

    Returns:
      An updated dictionary containing the tensor attributes mapped to the
        tensor name with the emem tensors(if any) staged in 'hbm.
    """
    if inputs:
      tensors = copy.deepcopy(
          self.inputs
          if self.collected_inputs is None
          else self.collected_inputs
      )
      non_collected_tensors = self.inputs
    else:
      tensors = copy.deepcopy(
          self.outputs
          if self.collected_outputs is None
          else self.collected_outputs
      )
      non_collected_tensors = self.outputs

    hbm_staging_tensors = {}
    for k, v in tensors.items():
      if v.storage_location == common_types.MemoryType.EMEM:
        # If the tensor is in EMEM, create a new tensor in HBM to account for
        # space needed for staging. This can be either a collected tensor or a
        # non-collected tensor.
        v_copy = copy.deepcopy(v)
        v_copy.storage_location = common_types.MemoryType.HBM
        hbm_staging_tensors[k + 'hbm'] = v_copy

        # EMEM is not involved in collectives, so set the EMEM tensor to the
        # tensor prior to any collectives.
        tensors[k] = copy.deepcopy(non_collected_tensors[k])
    tensors.update(hbm_staging_tensors)
    return tensors

  def pipeline_bytes_per_device(self) -> float:
    """Returns the number of bytes to send per device for pipeline parallelism."""
    if self.pipeline_config == sharding_types.PipelineConfig.NONE:
      return 0.0
    else:
      mbytes = 0.0
      for _, tensor in self.outputs.items():
        mbytes += tensor.bytes()
      return mbytes

  def initialize_collectives(self):
    """Initialize collectives, if needed, for sharded einsum."""
    self.collected_inputs, self.collected_outputs = (
        collectives.EinsumCollectives(self.inputs, self.outputs)
    ).generate_collectives()
