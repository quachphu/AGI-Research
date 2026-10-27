    ssi = Summary(value=[Summary.Value(tag=SESSION_START_INFO_TAG, metadata=smd)])

    mts = [MetricInfo(name=MetricName(tag=k)) for k in metric_dict]

    exp = Experiment(hparam_infos=hps, metric_infos=mts)

    content = HParamsPluginData(experiment=exp, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )
    exp = Summary(value=[Summary.Value(tag=EXPERIMENT_TAG, metadata=smd)])

    sei = SessionEndInfo(status=Status.Value("STATUS_SUCCESS"))
    content = HParamsPluginData(session_end_info=sei, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )
    sei = Summary(value=[Summary.Value(tag=SESSION_END_INFO_TAG, metadata=smd)])

    return exp, ssi, sei


def scalar(name, tensor, collections=None, new_style=False, double_precision=False):
    """Output a `Summary` protocol buffer containing a single scalar value.

    The generated Summary has a Tensor.proto containing the input Tensor.
    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A real numeric Tensor containing a single value.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
      new_style: Whether to use new style (tensor field) or old style (simple_value
        field). New style could lead to faster data loading.
    Returns:
      A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
    Raises:
      ValueError: If tensor has the wrong shape or type.
    """
    tensor = make_np(tensor).squeeze()
    if tensor.ndim != 0:
        raise AssertionError(f"Tensor should contain one element (0 dimensions). \
            Was given size: {tensor.size} and {tensor.ndim} dimensions.")
    # python float is double precision in numpy
    scalar = float(tensor)
    if new_style:
        tensor_proto = TensorProto(float_val=[scalar], dtype="DT_FLOAT")
        if double_precision:
            tensor_proto = TensorProto(double_val=[scalar], dtype="DT_DOUBLE")

        plugin_data = SummaryMetadata.PluginData(plugin_name="scalars")
        smd = SummaryMetadata(plugin_data=plugin_data)
        return Summary(
            value=[
                Summary.Value(
                    tag=name,
                    tensor=tensor_proto,
                    metadata=smd,
                )
            ]
        )
    else:
        return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])


def tensor_proto(tag, tensor):
    """Outputs a `Summary` protocol buffer containing the full tensor.
    The generated Summary has a Tensor.proto containing the input Tensor.
    Args:
      tag: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: Tensor to be converted to protobuf
    Returns:
      A tensor protobuf in a `Summary` protobuf.
    Raises:
      ValueError: If tensor is too big to be converted to protobuf, or
                     tensor data type is not supported
    """
    if tensor.numel() * tensor.itemsize >= (1 << 31):
        raise ValueError(
            "tensor is bigger than protocol buffer's hard limit of 2GB in size"
        )

    if tensor.dtype in _TENSOR_TYPE_MAP:
        dtype, field_name, conversion_fn = _TENSOR_TYPE_MAP[tensor.dtype]
        tensor_proto = TensorProto(
            **{
                "dtype": dtype,
                "tensor_shape": TensorShapeProto(
                    dim=[TensorShapeProto.Dim(size=x) for x in tensor.shape]
                ),
                field_name: conversion_fn(tensor),
            },
        )
    else:
        raise ValueError(f"{tag} has unsupported tensor dtype {tensor.dtype}")

    plugin_data = SummaryMetadata.PluginData(plugin_name="tensor")
    smd = SummaryMetadata(plugin_data=plugin_data)
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor_proto)])


def histogram_raw(name, min, max, num, sum, sum_squares, bucket_limits, bucket_counts):
    # pylint: disable=line-too-long
    """Output a `Summary` protocol buffer with a histogram.

    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      min: A float or int min value
      max: A float or int max value
      num: Int number of values
      sum: Float or int sum of all values
      sum_squares: Float or int sum of squares for all values
      bucket_limits: A numeric `Tensor` with upper value per bucket
      bucket_counts: A numeric `Tensor` with number of values per bucket
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    hist = HistogramProto(
        min=min,
        max=max,
        num=num,
        sum=sum,
        sum_squares=sum_squares,
        bucket_limit=bucket_limits,
        bucket=bucket_counts,
    )
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def histogram(name, values, bins, max_bins=None):
    # pylint: disable=line-too-long
    """Output a `Summary` protocol buffer with a histogram.

    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    This op reports an `InvalidArgument` error if any value is not finite.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      values: A real numeric `Tensor`. Any shape. Values to use to
        build the histogram.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    values = make_np(values)
    hist = make_histogram(values.astype(float), bins, max_bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram(values, bins, max_bins=None):
    """Convert values into a histogram proto using logic from histogram.cc."""
    if values.size == 0:
        raise ValueError("The input has no element.")
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            # pyrefly: ignore [no-matching-overload]
            counts = np.pad(
                counts,
                pad_width=[[0, subsampling - subsampling_remainder]],
                mode="constant",
                constant_values=0,
            )
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits

    # Find the first and the last bin defining the support of the histogram:

    cum_counts = np.cumsum(np.greater(counts, 0))
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
    start = int(start)
    end = int(end) + 1
    del cum_counts

    # TensorBoard only includes the right bin limits. To still have the leftmost limit
    # included, we include an empty bin left.
    # If start == 0, we need to add an empty one left, otherwise we can just include the bin left to the
    # first nonzero-count bin:
    counts = (
        counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
    )
    limits = limits[start : end + 1]

    if counts.size == 0 or limits.size == 0:
        raise ValueError("The histogram is empty, please file a bug report.")

    sum_sq = values.dot(values)
    return HistogramProto(
        min=values.min(),
        max=values.max(),
        num=len(values),
        sum=values.sum(),
        sum_squares=sum_sq,
        bucket_limit=limits.tolist(),
        bucket=counts.tolist(),
    )


def image(tag, tensor, rescale=1, dataformats="NCHW"):
    """Output a `Summary` protocol buffer with images.

    The summary has up to `max_images` summary values containing images. The
    images are built from `tensor` which must be 3-D with shape `[height, width,
    channels]` and where `channels` can be:
    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.
    The `name` in the outputted Summary.Value protobufs is generated based on the
    name, with a suffix depending on the max_outputs setting:
    *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
    *  If `max_outputs` is greater than 1, the summary value tags are
       generated sequentially as '*name*/image/0', '*name*/image/1', etc.
    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
        channels]` where `channels` is 1, 3, or 4.
        'tensor' can either have values in [0, 1] (float32) or [0, 255] (uint8).
        The image() function will scale the image values to [0, 255] by applying
        a scale factor of either 1 (uint8) or 255 (float32). Out-of-range values
        will be clipped.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tensor = make_np(tensor)
    tensor = convert_to_HWC(tensor, dataformats)
    # Do not assume that user passes in values in [0, 255], use data type to detect
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).clip(0, 255).astype(np.uint8)
    image = make_image(tensor, rescale=rescale)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def image_boxes(
    tag, tensor_image, tensor_boxes, rescale=1, dataformats="CHW", labels=None
):
    """Output a `Summary` protocol buffer with images."""
    tensor_image = make_np(tensor_image)
    tensor_image = convert_to_HWC(tensor_image, dataformats)
    tensor_boxes = make_np(tensor_boxes)
    tensor_image = tensor_image.astype(np.float32) * _calc_scale_factor(tensor_image)
    image = make_image(
        tensor_image.clip(0, 255).astype(np.uint8),
        rescale=rescale,
        rois=tensor_boxes,
        labels=labels,
    )
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def draw_boxes(disp_image, boxes, labels=None):
    # xyxy format
    num_boxes = boxes.shape[0]
    list_gt = range(num_boxes)
    for i in list_gt:
        disp_image = _draw_single_box(
            disp_image,
            boxes[i, 0],
            boxes[i, 1],
            boxes[i, 2],
            boxes[i, 3],
            display_str=None if labels is None else labels[i],
            color="Red",
        )
    return disp_image


def make_image(tensor, rescale=1, rois=None, labels=None):
    """Convert a numpy representation of an image to Image protobuf."""
    from PIL import Image

    height, width, channel = tensor.shape
    scaled_height = int(height * rescale)
    scaled_width = int(width * rescale)
    image = Image.fromarray(tensor)
    if rois is not None:
        image = draw_boxes(image, rois, labels=labels)
    ANTIALIAS = Image.Resampling.LANCZOS
    image = image.resize((scaled_width, scaled_height), ANTIALIAS)
    import io

    output = io.BytesIO()
    image.save(output, format="PNG")
    image_string = output.getvalue()
    output.close()
    return Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string,
    )


def video(tag, tensor, fps=4):
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)
    # If user passes in uint8, then we don't need to rescale by 255
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).clip(0, 255).astype(np.uint8)
    video = make_video(tensor, fps)
    return Summary(value=[Summary.Value(tag=tag, image=video)])


def make_video(tensor, fps):
    try:
        import moviepy  # noqa: F401
    except ImportError:
        print("add_video needs package moviepy")
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print(
            "moviepy is installed, but can't import moviepy.editor.",
            "Some packages could be missing [imageio, requests]",
        )
        return
    import tempfile

    _t, h, w, c = tensor.shape

    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    with tempfile.NamedTemporaryFile(suffix=".gif") as f:
        filename = f.name
        try:  # newer version of moviepy use logger instead of progress_bar argument.
            clip.write_gif(filename, verbose=False, logger=None)
        except TypeError:
            try:  # older version of moviepy does not support progress_bar argument.
                clip.write_gif(filename, verbose=False, progress_bar=False)
            except TypeError:
                clip.write_gif(filename, verbose=False)

        f.seek(0)
        tensor_string = f.read()

        return Summary.Image(
            height=h, width=w, colorspace=c, encoded_image_string=tensor_string
        )


def audio(tag, tensor, sample_rate=44100):
    array = make_np(tensor)
    array = array.squeeze()
    if abs(array).max() > 1:
        print("warning: audio amplitude out of range, auto clipped.")
        array = array.clip(-1, 1)
    if array.ndim != 1:
        raise AssertionError("input tensor should be 1 dimensional.")
    # pyrefly: ignore [no-matching-overload]
    array = (array * np.iinfo(np.int16).max).astype("<i2")

    import io
    import wave

    fio = io.BytesIO()
    with wave.open(fio, "wb") as wave_write:
        wave_write.setnchannels(1)
        wave_write.setsampwidth(2)
        wave_write.setframerate(sample_rate)
        wave_write.writeframes(array.data)
    audio_string = fio.getvalue()
    fio.close()
    audio = Summary.Audio(
        sample_rate=sample_rate,
        num_channels=1,
        length_frames=array.shape[-1],
        encoded_audio_string=audio_string,
        content_type="audio/wav",
    )
    return Summary(value=[Summary.Value(tag=tag, audio=audio)])


def custom_scalars(layout):
    categories = []
    for k, v in layout.items():
        charts = []
        for chart_name, chart_metadata in v.items():
            tags = chart_metadata[1]
            if chart_metadata[0] == "Margin":
                if len(tags) != 3:
                    raise AssertionError("len(tags) != 3")
                mgcc = layout_pb2.MarginChartContent(
                    series=[
                        layout_pb2.MarginChartContent.Series(
                            value=tags[0], lower=tags[1], upper=tags[2]
                        )
                    ]
                )
                chart = layout_pb2.Chart(title=chart_name, margin=mgcc)
            else:
                mlcc = layout_pb2.MultilineChartContent(tag=tags)
                chart = layout_pb2.Chart(title=chart_name, multiline=mlcc)
            charts.append(chart)
        categories.append(layout_pb2.Category(title=k, chart=charts))

    layout = layout_pb2.Layout(category=categories)
    plugin_data = SummaryMetadata.PluginData(plugin_name="custom_scalars")
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(
        dtype="DT_STRING",
        string_val=[layout.SerializeToString()],
        tensor_shape=TensorShapeProto(),
    )
    return Summary(
        value=[
            Summary.Value(tag="custom_scalars__config__", tensor=tensor, metadata=smd)
        ]
    )


def text(tag, text):
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="text", content=TextPluginData(version=0).SerializeToString()
    )
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(
        dtype="DT_STRING",
        string_val=[text.encode(encoding="utf_8")],
        tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
    )
    return Summary(
        value=[Summary.Value(tag=tag + "/text_summary", metadata=smd, tensor=tensor)]
    )


def pr_curve_raw(
    tag, tp, fp, tn, fn, precision, recall, num_thresholds=127, weights=None
):
    if num_thresholds > 127:  # weird, value > 127 breaks protobuf
        num_thresholds = 127
    data = np.stack((tp, fp, tn, fn, precision, recall))
    pr_curve_plugin_data = PrCurvePluginData(
        version=0, num_thresholds=num_thresholds
    ).SerializeToString()
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="pr_curves", content=pr_curve_plugin_data
    )
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=data.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=data.shape[0]),
                TensorShapeProto.Dim(size=data.shape[1]),
            ]
        ),
    )
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


def pr_curve(tag, labels, predictions, num_thresholds=127, weights=None):
    # weird, value > 127 breaks protobuf
    num_thresholds = min(num_thresholds, 127)
    data = compute_curve(
        labels, predictions, num_thresholds=num_thresholds, weights=weights
    )
    pr_curve_plugin_data = PrCurvePluginData(
        version=0, num_thresholds=num_thresholds
    ).SerializeToString()
    plugin_data = SummaryMetadata.PluginData(
        plugin_name="pr_curves", content=pr_curve_plugin_data
    )
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=data.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=data.shape[0]),
                TensorShapeProto.Dim(size=data.shape[1]),
            ]
        ),
    )
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


# https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/summary.py
def compute_curve(labels, predictions, num_thresholds=None, weights=None):
    _MINIMUM_COUNT = 1e-7

    if weights is None:
        weights = 1.0

    # Compute bins of true positives and false positives.
    # pyrefly: ignore [unsupported-operation]
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float64)
    # pyrefly: ignore [unsupported-operation]
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(
        bucket_indices,
        # pyrefly: ignore [bad-argument-type]
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights,
    )
    fp_buckets, _ = np.histogram(
        bucket_indices,
        # pyrefly: ignore [bad-argument-type]
        bins=num_thresholds,
        range=histogram_range,
        weights=(1.0 - float_labels) * weights,
    )

    # Obtain the reverse cumulative sum.
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
    return np.stack((tp, fp, tn, fn, precision, recall))


def _get_tensor_summary(
    name, display_name, description, tensor, content_type, components, json_config
):
    """Create a tensor summary with summary metadata.

    Args:
      name: Uniquely identifiable name of the summary op. Could be replaced by
        combination of name and type to make it unique even outside of this
        summary.
      display_name: Will be used as the display name in TensorBoard.
        Defaults to `name`.
      description: A longform readable description of the summary data. Markdown
        is supported.
      tensor: Tensor to display in summary.
      content_type: Type of content inside the Tensor.
      components: Bitmask representing present parts (vertices, colors, etc.) that
        belong to the summary.
      json_config: A string, JSON-serialized dictionary of ThreeJS classes
        configuration.

    Returns:
      Tensor summary with metadata.
    """
    import torch
    from tensorboard.plugins.mesh import metadata

    tensor = torch.as_tensor(tensor)

    tensor_metadata = metadata.create_summary_metadata(
        name,
        display_name,
        content_type,
        components,
        tensor.shape,
        description,
        json_config=json_config,
    )

    tensor = TensorProto(
        dtype="DT_FLOAT",
        float_val=tensor.reshape(-1).tolist(),
        tensor_shape=TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=tensor.shape[0]),
                TensorShapeProto.Dim(size=tensor.shape[1]),
                TensorShapeProto.Dim(size=tensor.shape[2]),
            ]
        ),
    )

    tensor_summary = Summary.Value(
        tag=metadata.get_instance_name(name, content_type),
        tensor=tensor,
        metadata=tensor_metadata,
    )

    return tensor_summary


def _get_json_config(config_dict):
    """Parse and returns JSON string from python dictionary."""
    json_config = "{}"
    if config_dict is not None:
        json_config = json.dumps(config_dict, sort_keys=True)
    return json_config


# https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/mesh/summary.py
def mesh(
    tag, vertices, colors, faces, config_dict, display_name=None, description=None
):
    """Output a merged `Summary` protocol buffer with a mesh/point cloud.

    Args:
      tag: A name for this summary operation.
      vertices: Tensor of shape `[dim_1, ..., dim_n, 3]` representing the 3D
        coordinates of vertices.
      faces: Tensor of shape `[dim_1, ..., dim_n, 3]` containing indices of
        vertices within each triangle.
      colors: Tensor of shape `[dim_1, ..., dim_n, 3]` containing colors for each
        vertex.
      display_name: If set, will be used as the display name in TensorBoard.
        Defaults to `name`.
      description: A longform readable description of the summary data. Markdown
        is supported.
      config_dict: Dictionary with ThreeJS classes names and configuration.

    Returns:
      Merged summary for mesh/point cloud representation.
    """
    from tensorboard.plugins.mesh import metadata
    from tensorboard.plugins.mesh.plugin_data_pb2 import MeshPluginData

    json_config = _get_json_config(config_dict)

    summaries = []
    tensors = [
        (vertices, MeshPluginData.VERTEX),
        (faces, MeshPluginData.FACE),
        (colors, MeshPluginData.COLOR),
    ]
    tensors = [tensor for tensor in tensors if tensor[0] is not None]
    components = metadata.get_components_bitmask(
        [content_type for (tensor, content_type) in tensors]
    )

    for tensor, content_type in tensors:
        summaries.append(
            _get_tensor_summary(
                tag,
                display_name,
                description,
                tensor,
                content_type,
                components,
                json_config,
            )
        )

    return Summary(value=summaries)
<|endoftext|># mypy: allow-untyped-defs
import numpy as np
import numpy.typing as npt


# Functions for converting
def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figures (matplotlib.pyplot.figure or list of figures): figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as plt_backend_agg

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data: npt.NDArray = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image


def _prepare_video(V):
    """
    Convert a 5D tensor into 4D tensor.

    Conversion is done from [batchsize, time(frame), channel(color), height, width]  (5D tensor)
    to [time(frame), new_width, new_height, channel] (4D tensor).

    A batch of images are spread to a grid, which forms a frame.
    e.g. Video with batchsize 16 will have a 4x4 grid.
    """
    b, t, c, h, w = V.shape

    if V.dtype == np.uint8:
        V = np.float32(V) / 255.0

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    # pad to nearest power of 2, all at once

    if not is_power2(V.shape[0]):

        len_addition = int(2 ** V.shape[0].bit_length() - V.shape[0])
        V = np.concatenate((V, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

    n_rows = 2 ** ((b.bit_length() - 1) // 2)

    n_cols = V.shape[0] // n_rows

    V = np.reshape(V, (n_rows, n_cols, t, c, h, w))
    V = np.transpose(V, axes=(2, 0, 4, 1, 5, 3))
    V = np.reshape(V, (t, n_rows * h, n_cols * w, c))

    return V


def make_grid(I, ncols=8):
    # I: N1HW or N3HW
    if not isinstance(I, np.ndarray):
        raise AssertionError("plugin error, should pass numpy array here")
    if I.shape[1] == 1:
        I = np.concatenate([I, I, I], 1)
    if I.ndim != 4 or I.shape[1] != 3:
        raise AssertionError("Input should be a 4D numpy array with 3 channels")
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((3, H * nrows, W * ncols), dtype=I.dtype)
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * H : (y + 1) * H, x * W : (x + 1) * W] = I[i]
            i = i + 1
    return canvas

    # if modality == 'IMG':
    #     if x.dtype == np.uint8:
    #         x = x.astype(np.float32) / 255.0


def convert_to_HWC(tensor, input_format):  # tensor: numpy array
    if len(set(input_format)) != len(input_format):
        raise AssertionError(f"You can not use the same dimension shorthand twice. \
            input_format: {input_format}")
    if len(tensor.shape) != len(input_format):
        raise AssertionError(f"size of input tensor and input format are different. \
        tensor shape: {tensor.shape}, input_format: {input_format}")
    input_format = input_format.upper()

    if len(input_format) == 4:
        index = [input_format.find(c) for c in "NCHW"]
        tensor_NCHW = tensor.transpose(index)
        tensor_CHW = make_grid(tensor_NCHW)
        return tensor_CHW.transpose(1, 2, 0)

    if len(input_format) == 3:
        index = [input_format.find(c) for c in "HWC"]
        tensor_HWC = tensor.transpose(index)
        if tensor_HWC.shape[2] == 1:
            tensor_HWC = np.concatenate([tensor_HWC, tensor_HWC, tensor_HWC], 2)
        return tensor_HWC

    if len(input_format) == 2:
        index = [input_format.find(c) for c in "HW"]
        tensor = tensor.transpose(index)
        tensor = np.stack([tensor, tensor, tensor], 2)
        return tensor
<|endoftext|># mypy: allow-untyped-defs
"""Provide an API for writing protocol buffers to event files to be consumed by TensorBoard for visualization."""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from ._convert_np import make_np
from ._embedding import get_embedding_info, make_mat, make_sprite, make_tsv, write_pbtxt
from ._onnx_graph import load_onnx_graph
from ._pytorch_graph import graph
from ._utils import figure_to_image
from .summary import (
    audio,
    custom_scalars,
    histogram,
    histogram_raw,
    hparams,
    image,
    image_boxes,
    mesh,
    pr_curve,
    pr_curve_raw,
    scalar,
    tensor_proto,
    text,
    video,
)

__all__ = ["FileWriter", "SummaryWriter"]


class FileWriter:
    """Writes protocol buffers to event files to be consumed by TensorBoard.

    The `FileWriter` class provides a mechanism to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(self, log_dir, max_queue=10, flush_secs=120, filename_suffix="") -> None:
        """Create a `FileWriter` and an event file.

        On construction the writer creates a new event file in `log_dir`.
        The other arguments to the constructor control the asynchronous writes to
        the event file.

        Args:
          log_dir: A string. Directory where event file will be written.
          max_queue: Integer. Size of the queue for pending events and
            summaries before one of the 'add' calls forces a flush to disk.
            Default is ten items.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk. Default is every two minutes.
          filename_suffix: A string. Suffix added to all event filenames
            in the log_dir directory. More details on filename construction in
            tensorboard.summary.writer.event_file_writer.EventFileWriter.
        """
        # Sometimes PosixPath is passed in and we need to coerce it to
        # a string in all cases
        # TODO: See if we can remove this in the future if we are
        # actually the ones passing in a PosixPath
        log_dir = str(log_dir)
        self.event_writer = EventFileWriter(
            log_dir, max_queue, flush_secs, filename_suffix
        )

    def get_logdir(self):
        """Return the directory where event file will be written."""
        return self.event_writer.get_logdir()

    def add_event(self, event, step=None, walltime=None) -> None:
        """Add an event to the event file.

        Args:
          event: An `Event` protocol buffer.
          step: Number. Optional global step value for training process
            to record with the event.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            # Make sure step is converted from numpy or other formats
            # since protobuf might not convert depending on version
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(self, summary, global_step=None, walltime=None) -> None:
        """Add a `Summary` protocol buffer to the event file.

        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.

        Args:
          summary: A `Summary` protocol buffer.
          global_step: Number. Optional global step value for training process
            to record with the summary.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)

    def add_graph(self, graph_profile, walltime=None) -> None:
        """Add a `Graph` and step stats protocol buffer to the event file.

        Args:
          graph_profile: A `Graph` and step stats protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        graph = graph_profile[0]
        stepstats = graph_profile[1]
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

        trm = event_pb2.TaggedRunMetadata(
            tag="step1", run_metadata=stepstats.SerializeToString()
        )
        event = event_pb2.Event(tagged_run_metadata=trm)
        self.add_event(event, None, walltime)

    def add_onnx_graph(self, graph, walltime=None) -> None:
        """Add a `Graph` protocol buffer to the event file.

        Args:
          graph: A `Graph` protocol buffer.
          walltime: float. Optional walltime to override the default (current)
            _get_file_writerfrom time.time())
        """
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

    def flush(self) -> None:
        """Flushes the event file to disk.

        Call this method to make sure that all pending events have been written to
        disk.
        """
        self.event_writer.flush()

    def close(self) -> None:
        """Flushes the event file to disk and close the file.

        Call this method when you do not need the summary writer anymore.
        """
        self.event_writer.close()

    def reopen(self) -> None:
        """Reopens the EventFileWriter.

        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the EventFileWriter was not closed.
        """
        self.event_writer.reopen()


class SummaryWriter:
    """Writes entries directly to event files in the log_dir to be consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(
        self,
        log_dir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
    ) -> None:
        """Create a `SummaryWriter` that will write out events and summaries to the event file.

        Args:
            log_dir (str): Save directory location. Default is
              runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
              Use hierarchical folder structure to compare
              between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
              for each new experiment to compare across them.
            comment (str): Comment log_dir suffix appended to the default
              ``log_dir``. If ``log_dir`` is assigned, this argument has no effect.
            purge_step (int):
              When logging crashes at step :math:`T+X` and restarts at step :math:`T`,
              any events whose global_step larger or equal to :math:`T` will be
              purged and hidden from TensorBoard.
              Note that crashed and resumed experiments should have the same ``log_dir``.
            max_queue (int): Size of the queue for pending events and
              summaries before one of the 'add' calls forces a flush to disk.
              Default is ten items.
            flush_secs (int): How often, in seconds, to flush the
              pending events and summaries to disk. Default is every two minutes.
            filename_suffix (str): Suffix added to all event filenames in
              the log_dir directory. More details on filename construction in
              tensorboard.summary.writer.event_file_writer.EventFileWriter.

        Examples::

            from torch.utils.tensorboard import SummaryWriter

            # create a summary writer with automatically generated folder name.
            writer = SummaryWriter()
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

            # create a summary writer using the specified folder name.
            writer = SummaryWriter("my_experiment")
            # folder location: my_experiment

            # create a summary writer with comment appended.
            writer = SummaryWriter(comment="LR_0.1_BATCH_16")
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/

        """
        torch._C._log_api_usage_once("tensorboard.create.summarywriter")
        if not log_dir:
            import socket
            from datetime import datetime

            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            log_dir = os.path.join(
                "runs", current_time + "_" + socket.gethostname() + comment
            )
        self.log_dir = log_dir
        self.purge_step = purge_step
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix

        # Initialize the file writers, but they can be cleared out on close
        # and recreated later as needed.
        self.file_writer = self.all_writers = None
        self._get_file_writer()

        # Create default bins for histograms, see generate_testdata.py in tensorflow/tensorboard
        v = 1e-12
        buckets = []
        neg_buckets = []
        while v < 1e20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets

    def _get_file_writer(self):
        """Return the default FileWriter instance. Recreates it if closed."""
        if self.all_writers is None or self.file_writer is None:
            # pyrefly: ignore [bad-assignment]
            self.file_writer = FileWriter(
                self.log_dir, self.max_queue, self.flush_secs, self.filename_suffix
            )
            # pyrefly: ignore [bad-assignment, missing-attribute]
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            if self.purge_step is not None:
                most_recent_step = self.purge_step
                # pyrefly: ignore [missing-attribute]
                self.file_writer.add_event(
                    Event(step=most_recent_step, file_version="brain.Event:2")
                )
                # pyrefly: ignore [missing-attribute]
                self.file_writer.add_event(
                    Event(
                        step=most_recent_step,
                        session_log=SessionLog(status=SessionLog.START),
                    )
                )
                self.purge_step = None
        return self.file_writer

    def get_logdir(self):
        """Return the directory where event files will be written."""
        return self.log_dir

    def add_hparams(
        self,
        hparam_dict,
        metric_dict,
        hparam_domain_discrete=None,
        run_name=None,
        global_step=None,
    ) -> None:
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_dict (dict): Each key-value pair in the dictionary is the
              name of the hyper parameter and its corresponding value.
              The type of the value can be one of `bool`, `string`, `float`,
              `int`, or `None`.
            metric_dict (dict): Each key-value pair in the dictionary is the
              name of the metric and its corresponding value. Note that the key used
              here should be unique in the tensorboard record. Otherwise the value
              you added by ``add_scalar`` will be displayed in hparam plugin. In most
              cases, this is unwanted.
            hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
              contains names of the hyperparameters and all discrete values they can hold
            run_name (str): Name of the run, to be included as part of the logdir.
              If unspecified, will use current timestamp.
            global_step (int): Global step value to record

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

        Expected result:

        .. image:: _static/img/tensorboard/add_hparam.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        if not run_name:
            run_name = str(time.time())
        logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp, global_step)
            w_hp.file_writer.add_summary(ssi, global_step)
            w_hp.file_writer.add_summary(sei, global_step)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v, global_step)

    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
        double_precision=False,
    ) -> None:
        """Add scalar data to summary.

        Args:
            tag (str): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              with seconds after epoch of event
            new_style (boolean): Whether to use new style (tensor field) or old
              style (simple_value field). New style could lead to faster data loading.
        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_scalar.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalar")

        summary = scalar(
            tag, scalar_value, new_style=new_style, double_precision=double_precision
        )
        self._get_file_writer().add_summary(summary, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None) -> None:
        """Add many scalar data to summary.

        Args:
            main_tag (str): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            r = 5
            for i in range(100):
                writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                                'xcosx':i*np.cos(i/r),
                                                'tanx': np.tan(i/r)}, i)
            writer.close()
            # This call adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.

        Expected result:

        .. image:: _static/img/tensorboard/add_scalars.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalars")
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self._get_file_writer().get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + main_tag.replace("/", "_") + "_" + tag
            if self.all_writers is None:
                raise AssertionError("self.all_writers is None")
            if fw_tag in self.all_writers:
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(
                    fw_tag, self.max_queue, self.flush_secs, self.filename_suffix
                )
                self.all_writers[fw_tag] = fw
            fw.add_summary(scalar(main_tag, scalar_value), global_step, walltime)

    def add_tensor(
        self,
        tag,
        tensor,
        global_step=None,
        walltime=None,
    ) -> None:
        """Add tensor data to summary.

        Args:
            tag (str): Data identifier
            tensor (torch.Tensor): tensor to save
            global_step (int): Global step value to record
        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = torch.tensor([1,2,3])
            writer.add_scalar('x', x)
            writer.close()

        Expected result:
            Summary::tensor::float_val [1,2,3]
                   ::tensor::shape [3]
                   ::tag 'x'

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_tensor")

        summary = tensor_proto(tag, tensor)
        self._get_file_writer().add_summary(summary, global_step, walltime)

    def add_histogram(
        self,
        tag,
        values,
        global_step=None,
        bins="tensorflow",
        walltime=None,
        max_bins=None,
    ) -> None:
        """Add histogram to summary.

        Args:
            tag (str): Data identifier
            values (torch.Tensor, numpy.ndarray, or string/blobname): Values to build histogram
            global_step (int): Global step value to record
            bins (str): One of {'tensorflow','auto', 'fd', ...}. This determines how the bins are made. You can find
              other options in: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            writer = SummaryWriter()
            for i in range(10):
                x = np.random.random(1000)
                writer.add_histogram('distribution centers', x + i, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_histogram.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
        if isinstance(bins, str) and bins == "tensorflow":
            bins = self.default_bins
        self._get_file_writer().add_summary(
            histogram(tag, values, bins, max_bins=max_bins), global_step, walltime
        )

    def add_histogram_raw(
        self,
        tag,
        min,
        max,
        num,
        sum,
        sum_squares,
        bucket_limits,
        bucket_counts,
        global_step=None,
        walltime=None,
    ) -> None:
        """Add histogram with raw data.

        Args:
            tag (str): Data identifier
            min (float or int): Min value
            max (float or int): Max value
            num (int): Number of values
            sum (float or int): Sum of all values
            sum_squares (float or int): Sum of squares for all values
            bucket_limits (torch.Tensor, numpy.ndarray): Upper value per bucket.
              The number of elements of it should be the same as `bucket_counts`.
            bucket_counts (torch.Tensor, numpy.ndarray): Number of values per bucket
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/histogram/README.md

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            writer = SummaryWriter()
            dummy_data = []
            for idx, value in enumerate(range(50)):
                dummy_data += [idx + 0.001] * value

            bins = list(range(50+2))
            bins = np.array(bins)
            values = np.array(dummy_data).astype(float).reshape(-1)
            counts, limits = np.histogram(values, bins=bins)
            sum_sq = values.dot(values)
            writer.add_histogram_raw(
                tag='histogram_with_raw_data',
                min=values.min(),
                max=values.max(),
                num=len(values),
                sum=values.sum(),
                sum_squares=sum_sq,
                bucket_limits=limits[1:].tolist(),
                bucket_counts=counts.tolist(),
                global_step=0)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_histogram_raw.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_histogram_raw")
        if len(bucket_limits) != len(bucket_counts):
            raise ValueError(
                "len(bucket_limits) != len(bucket_counts), see the document."
            )
        self._get_file_writer().add_summary(
            histogram_raw(
                tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts
            ),
            global_step,
            walltime,
        )

    def add_image(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"
    ) -> None:
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (str): Data identifier
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            dataformats (str): Image data format specification of the form
              CHW, HWC, HW, WH, etc.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            img = np.zeros((3, 100, 100))
            img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

            img_HWC = np.zeros((100, 100, 3))
            img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

            writer = SummaryWriter()
            writer.add_image('my_image', img, 0)

            # If you have non-default dimension setting, set the dataformats argument.
            writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_image.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_image")
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime
        )

    def add_images(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="NCHW"
    ) -> None:
        """Add batched image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (str): Data identifier
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            dataformats (str): Image data format specification of the form
              NCHW, NHWC, CHW, HWC, HW, WH, etc.
        Shape:
            img_tensor: Default is :math:`(N, 3, H, W)`. If ``dataformats`` is specified, other shape will be
            accepted. e.g. NCHW or NHWC.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np

            img_batch = np.zeros((16, 3, 100, 100))
            for i in range(16):
                img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
                img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

            writer = SummaryWriter()
            writer.add_images('my_image_batch', img_batch, 0)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_images.png
           :scale: 30 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_images")
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime
        )

    def add_image_with_boxes(
        self,
        tag,
        img_tensor,
        box_tensor,
        global_step=None,
        walltime=None,
        rescale=1,
        dataformats="CHW",
        labels=None,
    ) -> None:
        """Add image and draw bounding boxes on the image.

        Args:
            tag (str): Data identifier
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
            box_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Box data (for detected objects)
              box should be represented as [x1, y1, x2, y2].
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            rescale (float): Optional scale override
            dataformats (str): Image data format specification of the form
              NCHW, NHWC, CHW, HWC, HW, WH, etc.
            labels (list of string): The label to be shown for each bounding box.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. It can be specified with ``dataformats`` argument.
            e.g. CHW or HWC

            box_tensor: (torch.Tensor, numpy.ndarray, or string/blobname): NX4,  where N is the number of
            boxes and each 4 elements in a row represents (xmin, ymin, xmax, ymax).
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_image_with_boxes")
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            if len(labels) != box_tensor.shape[0]:
                labels = None
        self._get_file_writer().add_summary(
            image_boxes(
                tag,
                img_tensor,
                box_tensor,
                rescale=rescale,
                dataformats=dataformats,
                labels=labels,
            ),
            global_step,
            walltime,
        )

    def add_figure(
        self,
        tag: str,
        figure: Figure | list[Figure],
        global_step: int | None = None,
        close: bool = True,
        walltime: float | None = None,
    ) -> None:
        """Render matplotlib figure into an image and add it to summary.

        Note that this requires the ``matplotlib`` package.

        Args:
            tag: Data identifier
            figure: Figure or a list of figures
            global_step: Global step value to record
            close: Flag to automatically close the figure
            walltime: Optional override default walltime (time.time())
              seconds after epoch of event
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_figure")
        if isinstance(figure, list):
            self.add_image(
                tag,
                figure_to_image(figure, close),
                global_step,
                walltime,
                dataformats="NCHW",
            )
        else:
            self.add_image(
                tag,
                figure_to_image(figure, close),
                global_step,
                walltime,
                dataformats="CHW",
            )

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None) -> None:
        """Add video data to summary.

        Note that this requires the ``moviepy`` package.

        Args:
            tag (str): Data identifier
            vid_tensor (torch.Tensor): Video data
            global_step (int): Global step value to record
            fps (float or int): Frames per second
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_video")
        self._get_file_writer().add_summary(
            video(tag, vid_tensor, fps), global_step, walltime
        )

    def add_audio(
        self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None
    ) -> None:
        """Add audio data to summary.

        Args:
            tag (str): Data identifier
            snd_tensor (torch.Tensor): Sound data
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_audio")
        self._get_file_writer().add_summary(
            audio(tag, snd_tensor, sample_rate=sample_rate), global_step, walltime
        )

    def add_text(self, tag, text_string, global_step=None, walltime=None) -> None:
        """Add text data to summary.

        Args:
            tag (str): Data identifier
            text_string (str): String to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Examples::

            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_text")
        self._get_file_writer().add_summary(
            text(tag, text_string), global_step, walltime
        )

    def add_onnx_graph(self, prototxt) -> None:
        torch._C._log_api_usage_once("tensorboard.logging.add_onnx_graph")
        self._get_file_writer().add_onnx_graph(load_onnx_graph(prototxt))

    def add_graph(
        self, model, input_to_model=None, verbose=False, use_strict_trace=True
    ) -> None:
        """Add graph data to summary.

        Args:
            model (torch.nn.Module): Model to draw.
            input_to_model (torch.Tensor or list of torch.Tensor): A variable or a tuple of
                variables to be fed.
            verbose (bool): Whether to print graph structure in console.
            use_strict_trace (bool): Whether to pass keyword argument `strict` to
                `torch.jit.trace`. Pass False when you want the tracer to
                record your mutable container types (list, dict)
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_graph")
        # A valid PyTorch model should have a 'forward' method
        self._get_file_writer().add_graph(
            graph(model, input_to_model, verbose, use_strict_trace)
        )

    @staticmethod
    def _encode(rawstr):
        # I'd use urllib but, I'm unsure about the differences from python3 to python2, etc.
        retval = rawstr
        retval = retval.replace("%", f"%{ord('%'):02x}")
        retval = retval.replace("/", f"%{ord('/'):02x}")
        retval = retval.replace("\\", "%%%02x" % (ord("\\")))  # noqa: UP031
        return retval

    def add_embedding(
        self,
        mat,
        metadata=None,
        label_img=None,
        global_step=None,
        tag="default",
        metadata_header=None,
    ) -> None:
        """Add embedding projector data to summary.

        Args:
            mat (torch.Tensor or numpy.ndarray): A matrix which each row is the feature vector of the data point
            metadata (list): A list of labels, each element will be converted to string
            label_img (torch.Tensor): Images correspond to each data point
            global_step (int): Global step value to record
            tag (str): Name for the embedding
            metadata_header (list): A list of headers for multi-column metadata. If given, each metadata must be
                a list with values corresponding to headers.
        Shape:
            mat: :math:`(N, D)`, where N is number of data and D is feature dimension

            label_img: :math:`(N, C, H, W)`

        Examples::

            import keyword
            import torch
            meta = []
            while len(meta)<100:
                meta = meta+keyword.kwlist # get some strings
            meta = meta[:100]

            for i, v in enumerate(meta):
                meta[i] = v+str(i)

            label_img = torch.rand(100, 3, 10, 32)
            for i in range(100):
                label_img[i]*=i/100.0

            writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), metadata=meta)

        .. note::
            Categorical (i.e. non-numeric) metadata cannot have more than 50 unique values if they are to be used for
            coloring in the embedding projector.

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_embedding")
        mat = make_np(mat)
        if global_step is None:
            global_step = 0
            # clear pbtxt?

        # Maybe we should encode the tag so slashes don't trip us up?
        # I don't think this will mess us up, but better safe than sorry.
        subdir = f"{str(global_step).zfill(5)}/{self._encode(tag)}"
        save_path = os.path.join(self._get_file_writer().get_logdir(), subdir)

        fs = tf.io.gfile
        if fs.exists(save_path):
            if fs.isdir(save_path):
                print(
                    "warning: Embedding dir exists, did you set global_step for add_embedding()?"
                )
            else:
                raise NotADirectoryError(
                    f"Path: `{save_path}` exists, but is a file. Cannot proceed."
                )
        else:
            fs.makedirs(save_path)

        if metadata is not None:
            if mat.shape[0] != len(
                metadata
            ):
                raise AssertionError("#labels should equal with #data points")
            make_tsv(metadata, save_path, metadata_header=metadata_header)

        if label_img is not None:
            if mat.shape[0] != label_img.shape[0]:
                raise AssertionError("#images should equal with #data points")
            make_sprite(label_img, save_path)

        if mat.ndim != 2:
            raise AssertionError("mat should be 2D, where mat.size(0) is the number of data points")
        make_mat(mat, save_path)

        # Filesystem doesn't necessarily have append semantics, so we store an
        # internal buffer to append to and re-write whole file after each
        # embedding is added
        if not hasattr(self, "_projector_config"):
            self._projector_config = ProjectorConfig()
        embedding_info = get_embedding_info(
            metadata, label_img, subdir, global_step, tag
        )
        self._projector_config.embeddings.extend([embedding_info])


        from google.protobuf import text_format

        config_pbtxt = text_format.MessageToString(self._projector_config)
        write_pbtxt(self._get_file_writer().get_logdir(), config_pbtxt)

    def add_pr_curve(
        self,
        tag,
        labels,
        predictions,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ) -> None:
        """Add precision recall curve.

        Plotting a precision-recall curve lets you understand your model's
        performance under different threshold settings. With this function,
        you provide the ground truth labeling (T/F) and prediction confidence
        (usually the output of your model) for each target. The TensorBoard UI
        will let you choose the threshold interactively.

        Args:
            tag (str): Data identifier
            labels (torch.Tensor, numpy.ndarray, or string/blobname):
              Ground truth data. Binary label for each element.
            predictions (torch.Tensor, numpy.ndarray, or string/blobname):
              The probability that an element be classified as true.
              Value should be in [0, 1]
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            labels = np.random.randint(2, size=100)  # binary label
            predictions = np.random.rand(100)
            writer = SummaryWriter()
            writer.add_pr_curve('pr_curve', labels, predictions, 0)
            writer.close()

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve")
        labels, predictions = make_np(labels), make_np(predictions)
        self._get_file_writer().add_summary(
            pr_curve(tag, labels, predictions, num_thresholds, weights),
            global_step,
            walltime,
        )

    def add_pr_curve_raw(
        self,
        tag,
        true_positive_counts,
        false_positive_counts,
        true_negative_counts,
        false_negative_counts,
        precision,
        recall,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ) -> None:
        """Add precision recall curve with raw data.

        Args:
            tag (str): Data identifier
            true_positive_counts (torch.Tensor, numpy.ndarray, or string/blobname): true positive counts
            false_positive_counts (torch.Tensor, numpy.ndarray, or string/blobname): false positive counts
            true_negative_counts (torch.Tensor, numpy.ndarray, or string/blobname): true negative counts
            false_negative_counts (torch.Tensor, numpy.ndarray, or string/blobname): false negative counts
            precision (torch.Tensor, numpy.ndarray, or string/blobname): precision
            recall (torch.Tensor, numpy.ndarray, or string/blobname): recall
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve_raw")
        self._get_file_writer().add_summary(
            pr_curve_raw(
                tag,
                true_positive_counts,
                false_positive_counts,
                true_negative_counts,
                false_negative_counts,
                precision,
                recall,
                num_thresholds,
                weights,
            ),
            global_step,
            walltime,
        )

    def add_custom_scalars_multilinechart(
        self, tags, category="default", title="untitled"
    ) -> None:
        """Shorthand for creating multilinechart. Similar to ``add_custom_scalars()``, but the only necessary argument is *tags*.

        Args:
            tags (list): list of tags that have been used in ``add_scalar()``

        Examples::

            writer.add_custom_scalars_multilinechart(['twse/0050', 'twse/2330'])
        """
        torch._C._log_api_usage_once(
            "tensorboard.logging.add_custom_scalars_multilinechart"
        )
        layout = {category: {title: ["Multiline", tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_custom_scalars_marginchart(
        self, tags, category="default", title="untitled"
    ) -> None:
        """Shorthand for creating marginchart.

        Similar to ``add_custom_scalars()``, but the only necessary argument is *tags*,
        which should have exactly 3 elements.

        Args:
            tags (list): list of tags that have been used in ``add_scalar()``

        Examples::

            writer.add_custom_scalars_marginchart(['twse/0050', 'twse/2330', 'twse/2006'])
        """
        torch._C._log_api_usage_once(
            "tensorboard.logging.add_custom_scalars_marginchart"
        )
        if len(tags) != 3:
            raise AssertionError(f"Expected 3 tags, got {len(tags)}.")
        layout = {category: {title: ["Margin", tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_custom_scalars(self, layout) -> None:
        """Create special chart by collecting charts tags in 'scalars'.

        NOTE: This function can only be called once for each SummaryWriter() object.

        Because it only provides metadata to tensorboard, the function can be called before or after the training loop.

        Args:
            layout (dict): {categoryName: *charts*}, where *charts* is also a dictionary
              {chartName: *ListOfProperties*}. The first element in *ListOfProperties* is the chart's type
              (one of **Multiline** or **Margin**) and the second element should be a list containing the tags
              you have used in add_scalar function, which will be collected into the new chart.

        Examples::

            layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
                         'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                              'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}

            writer.add_custom_scalars(layout)
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_custom_scalars")
        self._get_file_writer().add_summary(custom_scalars(layout))

    def add_mesh(
        self,
        tag,
        vertices,
        colors=None,
        faces=None,
        config_dict=None,
        global_step=None,
        walltime=None,
    ) -> None:
        """Add meshes or 3D point clouds to TensorBoard.

        The visualization is based on Three.js,
        so it allows users to interact with the rendered object. Besides the basic definitions
        such as vertices, faces, users can further provide camera parameter, lighting condition, etc.
        Please check https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene for
        advanced usage.

        Args:
            tag (str): Data identifier
            vertices (torch.Tensor): List of the 3D coordinates of vertices.
            colors (torch.Tensor): Colors for each vertex
            faces (torch.Tensor): Indices of vertices within each triangle. (Optional)
            config_dict: Dictionary with ThreeJS classes names and configuration.
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Shape:
            vertices: :math:`(B, N, 3)`. (batch, number_of_vertices, channels)

            colors: :math:`(B, N, 3)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.

            faces: :math:`(B, N, 3)`. The values should lie in [0, number_of_vertices] for type `uint8`.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            vertices_tensor = torch.as_tensor([
                [1, 1, 1],
                [-1, -1, 1],
                [1, -1, -1],
                [-1, 1, -1],
            ], dtype=torch.float).unsqueeze(0)
            colors_tensor = torch.as_tensor([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 255],
            ], dtype=torch.int).unsqueeze(0)
            faces_tensor = torch.as_tensor([
                [0, 2, 3],
                [0, 3, 1],
                [0, 1, 2],
                [1, 3, 2],
            ], dtype=torch.int).unsqueeze(0)

            writer = SummaryWriter()
            writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)

            writer.close()
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_mesh")
        self._get_file_writer().add_summary(
            mesh(tag, vertices, colors, faces, config_dict), global_step, walltime
        )

    def flush(self) -> None:
        """Flushes the event file to disk.

        Call this method to make sure that all pending events have been written to
        disk.
        """
        if self.all_writers is None:
            return
        for writer in self.all_writers.values():
            writer.flush()

    def close(self) -> None:
        if self.all_writers is None:
            return  # ignore double close
        for writer in self.all_writers.values():
            writer.flush()
            writer.close()
        # pyrefly: ignore [bad-assignment]
        self.file_writer = self.all_writers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
<|endoftext|>import torch

from collections.abc import Sequence
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto


def attr_value_proto(dtype: object, shape: Sequence[int] | None, s: str | None) -> dict[str, AttrValue]:
    """Create a dict of objects matching a NodeDef's attr field.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/attr_value.proto
    specifically designed for a NodeDef. The values have been reverse engineered from
    standard TensorBoard logged data.
    """
    attr = {}
    if s is not None:
        attr["attr"] = AttrValue(s=s.encode(encoding="utf_8"))
    if shape is not None:
        shapeproto = tensor_shape_proto(shape)
        attr["_output_shapes"] = AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))
    return attr


def tensor_shape_proto(outputsize: Sequence[int]) -> TensorShapeProto:
    """Create an object matching a tensor_shape field.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor_shape.proto .
    """
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in outputsize])


def node_proto(
    name: str,
    op: str = "UnSpecified",
    input: list[str] | str | None = None,
    dtype: torch.dtype | None = None,
    shape: tuple[int, ...] | None = None,
    outputsize: Sequence[int] | None = None,
    attributes: str = "",
) -> NodeDef:
    """Create an object matching a NodeDef.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/node_def.proto .
    """
    if input is None:
        input = []
    if not isinstance(input, list):
        input = [input]
    return NodeDef(
        name=name.encode(encoding="utf_8"),
        op=op,
        input=input,
        attr=attr_value_proto(dtype, outputsize, attributes),
    )
<|endoftext|># mypy: allow-untyped-defs
from torch._C import (
    _get_backcompat_broadcast_warn,
    _get_backcompat_keepdim_warn,
    _set_backcompat_broadcast_warn,
    _set_backcompat_keepdim_warn,
)


class Warning:
    def __init__(self, setter, getter) -> None:
        self.setter = setter
        self.getter = getter

    def set_enabled(self, value) -> None:
        self.setter(value)

    def get_enabled(self):
        return self.getter()

    enabled = property(get_enabled, set_enabled)


broadcast_warning = Warning(
    _set_backcompat_broadcast_warn, _get_backcompat_broadcast_warn
)
keepdim_warning = Warning(_set_backcompat_keepdim_warn, _get_backcompat_keepdim_warn)
<|endoftext|># mypy: allow-untyped-defs
import io
import pickle
import warnings
from collections.abc import Collection

from torch.utils._import_utils import dill_available
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


__all__ = ["traverse", "traverse_dps"]

DataPipe = IterDataPipe | MapDataPipe
# pyrefly: ignore [invalid-type-alias]
DataPipeGraph = dict[int, tuple[DataPipe, "DataPipeGraph"]]


def _stub_unpickler() -> str:
    return "STUB"


# TODO(VitalyFedyunin): Make sure it works without dill module installed
def _list_connected_datapipes(
    scan_obj: DataPipe, only_datapipe: bool, cache: set[int]
) -> list[DataPipe]:
    f = io.BytesIO()
    p = pickle.Pickler(
        f
    )  # Not going to work for lambdas, but dill infinite loops on typing and can't be used as is
    if dill_available():
        from dill import Pickler as dill_Pickler

        d = dill_Pickler(f)
    else:
        d = None

    captured_connections = []

    def getstate_hook(ori_state):
        state = None
        if isinstance(ori_state, dict):
            state = {}
            for k, v in ori_state.items():
                if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                    state[k] = v
        elif isinstance(ori_state, (tuple, list)):
            state = []  # type: ignore[assignment]
            for v in ori_state:
                if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                    state.append(v)  # type: ignore[attr-defined]
        elif isinstance(ori_state, (IterDataPipe, MapDataPipe, Collection)):
            state = ori_state  # type: ignore[assignment]
        return state

    def reduce_hook(obj):
        if obj == scan_obj or id(obj) in cache:
            raise NotImplementedError
        else:
            captured_connections.append(obj)
            # Adding id to remove duplicate DataPipe serialized at the same level
            cache.add(id(obj))
            return _stub_unpickler, ()

    datapipe_classes: tuple[type[DataPipe]] = (IterDataPipe, MapDataPipe)  # type: ignore[assignment]

    try:
        for cls in datapipe_classes:
            cls.set_reduce_ex_hook(reduce_hook)
            if only_datapipe:
                cls.set_getstate_hook(getstate_hook)
        try:
            p.dump(scan_obj)
        except (pickle.PickleError, AttributeError, TypeError):
            if dill_available():
                # pyrefly: ignore [missing-attribute]
                d.dump(scan_obj)
            else:
                raise
    finally:
        for cls in datapipe_classes:
            cls.set_reduce_ex_hook(None)
            if only_datapipe:
                cls.set_getstate_hook(None)
        if dill_available():
            from dill import extend as dill_extend

            dill_extend(False)  # Undo change to dispatch table
    return captured_connections


def traverse_dps(datapipe: DataPipe) -> DataPipeGraph:
    r"""
    Traverse the DataPipes and their attributes to extract the DataPipe graph.

    This only looks into the attribute from each DataPipe that is either a
    DataPipe and a Python collection object such as ``list``, ``tuple``,
    ``set`` and ``dict``.

    Args:
        datapipe: the end DataPipe of the graph
    Returns:
        A graph represented as a nested dictionary, where keys are ids of DataPipe instances
        and values are tuples of DataPipe instance and the sub-graph
    """
    cache: set[int] = set()
    return _traverse_helper(datapipe, only_datapipe=True, cache=cache)


def traverse(datapipe: DataPipe, only_datapipe: bool | None = None) -> DataPipeGraph:
    r"""
    Traverse the DataPipes and their attributes to extract the DataPipe graph.

    [Deprecated]
    When ``only_dataPipe`` is specified as ``True``, it would only look into the
    attribute from each DataPipe that is either a DataPipe and a Python collection object
    such as ``list``, ``tuple``, ``set`` and ``dict``.

    Note:
        This function is deprecated. Please use `traverse_dps` instead.

    Args:
        datapipe: the end DataPipe of the graph
        only_datapipe: If ``False`` (default), all attributes of each DataPipe are traversed.
          This argument is deprecated and will be removed after the next release.
    Returns:
        A graph represented as a nested dictionary, where keys are ids of DataPipe instances
        and values are tuples of DataPipe instance and the sub-graph
    """
    msg = (
        "`traverse` function is deprecated and will be removed after 1.13. "
        "Please use `traverse_dps` instead."
    )
    if not only_datapipe:
        msg += " And, the behavior will be changed to the equivalent of `only_datapipe=True`."
    warnings.warn(msg, FutureWarning, stacklevel=2)
    if only_datapipe is None:
        only_datapipe = False
    cache: set[int] = set()
    return _traverse_helper(datapipe, only_datapipe, cache)


# Add cache here to prevent infinite recursion on DataPipe
def _traverse_helper(
    datapipe: DataPipe, only_datapipe: bool, cache: set[int]
) -> DataPipeGraph:
    if not isinstance(datapipe, (IterDataPipe, MapDataPipe)):
        raise RuntimeError(
            f"Expected `IterDataPipe` or `MapDataPipe`, but {type(datapipe)} is found"
        )

    dp_id = id(datapipe)
    if dp_id in cache:
        return {}
    cache.add(dp_id)
    # Using cache.copy() here is to prevent the same DataPipe pollutes the cache on different paths
    items = _list_connected_datapipes(datapipe, only_datapipe, cache.copy())
    d: DataPipeGraph = {dp_id: (datapipe, {})}
    for item in items:
        # Using cache.copy() here is to prevent recursion on a single path rather than global graph
        # Single DataPipe can present multiple times in different paths in graph
        # pyrefly: ignore [no-matching-overload]
        d[dp_id][1].update(_traverse_helper(item, only_datapipe, cache.copy()))
    return d
<|endoftext|># mypy: allow-untyped-defs
import itertools
from collections.abc import Iterable, Iterator, Sequence, Sized
from typing import Generic, TypeVar

import torch


# Note: For benchmarking changes to samplers, see:
# /benchmarks/data/samplers_bench.py
# This benchmark compares the performance of different sampler implementations
# and can be used to evaluate the impact of optimizations.


__all__ = [
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]


_T_co = TypeVar("_T_co", covariant=True)


class Sampler(Generic[_T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements,
    and may provide a :meth:`__len__` method that returns the length of the returned iterators.

    Example:
        >>> # xdoctest: +SKIP
        >>> class AccedingSequenceLengthSampler(Sampler[int]):
        >>>     def __init__(self, data: List[str]) -> None:
        >>>         self.data = data
        >>>
        >>>     def __len__(self) -> int:
        >>>         return len(self.data)
        >>>
        >>>     def __iter__(self) -> Iterator[int]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         yield from torch.argsort(sizes).tolist()
        >>>
        >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
        >>>     def __init__(self, data: List[str], batch_size: int) -> None:
        >>>         self.data = data
        >>>         self.batch_size = batch_size
        >>>
        >>>     def __len__(self) -> int:
        >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
        >>>
        >>>     def __iter__(self) -> Iterator[List[int]]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
        >>>             yield batch.tolist()

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __iter__(self) -> Iterator[_T_co]:
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising a `NotImplementedError` will propagate and make the call fail
    #     where it could have used `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)


class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Sized): data source to sample from. Must implement __len__.
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Sized): data source to sample from. Must implement __len__.
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[
                : self.num_samples % n
            ]

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator).tolist():
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> list(
        ...     WeightedRandomSampler(
        ...         [0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True
        ...     )
        ... )
        [4, 4, 1, 4, 5]
        >>> list(
        ...     WeightedRandomSampler(
        ...         [0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False
        ...     )
        ... )
        [0, 1, 4, 3, 2]
    """

    weights: torch.Tensor
    num_samples: int
    replacement: bool

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator=None,
    ) -> None:
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[list[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(
        ...     BatchSampler(
        ...         SequentialSampler(range(10)), batch_size=3, drop_last=False
        ...     )
        ... )
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(
        ...     BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True)
        ... )
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        sampler_iter = iter(self.sampler)
        if self.drop_last:
            # Create multiple references to the same iterator
            args = [sampler_iter] * self.batch_size
            for batch_droplast in zip(*args, strict=False):
                yield [*batch_droplast]
        else:
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
<|endoftext|># mypy: allow-untyped-defs
import inspect
import warnings
from typing import Any
from typing_extensions import deprecated

import torch
from torch.utils.data.datapipes.iter.sharding import (
    _ShardingIterDataPipe,
    SHARDING_PRIORITIES,
)
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps


__all__ = [
    "apply_random_seed",
    "apply_sharding",
    "apply_shuffle_seed",
    "apply_shuffle_settings",
    "get_all_graph_pipes",
]


def get_all_graph_pipes(graph: DataPipeGraph) -> list[DataPipe]:
    return _get_all_graph_pipes_helper(graph, set())


def _get_all_graph_pipes_helper(
    graph: DataPipeGraph, id_cache: set[int]
) -> list[DataPipe]:
    results: list[DataPipe] = []
    for dp_id, (datapipe, sub_graph) in graph.items():
        if dp_id in id_cache:
            continue
        id_cache.add(dp_id)
        results.append(datapipe)
        results.extend(_get_all_graph_pipes_helper(sub_graph, id_cache))
    return results


def _is_sharding_datapipe(datapipe: DataPipe) -> bool:
    return isinstance(datapipe, _ShardingIterDataPipe) or (
        hasattr(datapipe, "apply_sharding")
        and inspect.ismethod(datapipe.apply_sharding)
    )


def apply_sharding(
    datapipe: DataPipe,
    num_of_instances: int,
    instance_id: int,
    sharding_group=SHARDING_PRIORITIES.DEFAULT,
) -> DataPipe:
    r"""
    Apply dynamic sharding over the ``sharding_filter`` DataPipe that has a method ``apply_sharding``.

    RuntimeError will be raised when multiple ``sharding_filter`` are presented in the same branch.
    """
    graph = traverse_dps(datapipe)

    def _helper(graph, prev_applied=None) -> None:
        for dp, sub_graph in graph.values():
            applied = None
            if _is_sharding_datapipe(dp):
                if prev_applied is not None:
                    raise RuntimeError(
                        "Sharding twice on a single pipeline is likely unintended and will cause data loss. "
                        f"Sharding already applied to {prev_applied} while trying to apply to {dp}"
                    )
                # For BC, only provide sharding_group if accepted
                sig = inspect.signature(dp.apply_sharding)
                if len(sig.parameters) < 3:
                    dp.apply_sharding(num_of_instances, instance_id)
                else:
                    dp.apply_sharding(
                        num_of_instances, instance_id, sharding_group=sharding_group
                    )
                applied = dp
            if applied is None:
                applied = prev_applied
            _helper(sub_graph, applied)

    _helper(graph)

    return datapipe


def _is_shuffle_datapipe(datapipe: DataPipe) -> bool:
    return (
        hasattr(datapipe, "set_shuffle")
        and hasattr(datapipe, "set_seed")
        and inspect.ismethod(datapipe.set_shuffle)
        and inspect.ismethod(datapipe.set_seed)
    )


def apply_shuffle_settings(datapipe: DataPipe, shuffle: bool | None = None) -> DataPipe:
    r"""
    Traverse the graph of ``DataPipes`` to find and set shuffle attribute.

    Apply the method to each `DataPipe` that has APIs of ``set_shuffle``
    and ``set_seed``.

    Args:
        datapipe: DataPipe that needs to set shuffle attribute
        shuffle: Shuffle option (default: ``None`` and no-op to the graph)
    """
    if shuffle is None:
        return datapipe

    graph = traverse_dps(datapipe)
    all_pipes = get_all_graph_pipes(graph)
    shufflers = [pipe for pipe in all_pipes if _is_shuffle_datapipe(pipe)]
    if not shufflers and shuffle:
        warnings.warn(
            "`shuffle=True` was set, but the datapipe does not contain a `Shuffler`. Adding one at the end. "
            "Be aware that the default buffer size might not be sufficient for your task.",
            stacklevel=2,
        )
        datapipe = datapipe.shuffle()
        shufflers = [
            datapipe,
        ]

    for shuffler in shufflers:
        shuffler.set_shuffle(shuffle)

    return datapipe


@deprecated(
    "`apply_shuffle_seed` is deprecated since 1.12 and will be removed in the future releases. "
    "Please use `apply_random_seed` instead.",
    category=FutureWarning,
)
def apply_shuffle_seed(datapipe: DataPipe, rng: Any) -> DataPipe:
    return apply_random_seed(datapipe, rng)


def _is_random_datapipe(datapipe: DataPipe) -> bool:
    return hasattr(datapipe, "set_seed") and inspect.ismethod(datapipe.set_seed)


def apply_random_seed(datapipe: DataPipe, rng: torch.Generator) -> DataPipe:
    r"""
    Traverse the graph of ``DataPipes`` to find random ``DataPipe`` with an API of ``set_seed``.

    Then set the random seed based on the provided RNG to those ``DataPipe``.

    Args:
        datapipe: DataPipe that needs to set randomness
        rng: Random number generator to generate random seeds
    """
    graph = traverse_dps(datapipe)
    all_pipes = get_all_graph_pipes(graph)
    # Using a set to track id of DataPipe to prevent setting randomness per DataPipe more than once.
    # And, `id` is used in case of unhashable DataPipe
    cache = set()
    random_datapipes = []
    for pipe in all_pipes:
        if id(pipe) in cache:
            continue
        if _is_random_datapipe(pipe):
            random_datapipes.append(pipe)
            cache.add(id(pipe))

    for pipe in random_datapipes:
        random_seed = int(
            torch.empty((), dtype=torch.int64).random_(generator=rng).item()
        )
        pipe.set_seed(random_seed)

    return datapipe
<|endoftext|>from torch.utils.data.dataloader import (
    _DatasetKind,
    DataLoader,
    default_collate,
    default_convert,
    get_worker_info,
)
from torch.utils.data.datapipes._decorator import (
    argument_validation,
    functional_datapipe,
    guaranteed_datapipes_determinism,
    non_deterministic,
    runtime_validation,
    runtime_validation_disabled,
)
from torch.utils.data.datapipes.datapipe import (
    DataChunk,
    DFIterDataPipe,
    IterDataPipe,
    MapDataPipe,
)
from torch.utils.data.dataset import (
    ChainDataset,
    ConcatDataset,
    Dataset,
    IterableDataset,
    random_split,
    StackDataset,
    Subset,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)


__all__ = [
    "BatchSampler",
    "ChainDataset",
    "ConcatDataset",
    "DFIterDataPipe",
    "DataChunk",
    "DataLoader",
    "Dataset",
    "DistributedSampler",
    "IterDataPipe",
    "IterableDataset",
    "MapDataPipe",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "StackDataset",
    "Subset",
    "SubsetRandomSampler",
    "TensorDataset",
    "WeightedRandomSampler",
    "_DatasetKind",
    "argument_validation",
    "default_collate",
    "default_convert",
    "functional_datapipe",
    "get_worker_info",
    "guaranteed_datapipes_determinism",
    "non_deterministic",
    "random_split",
    "runtime_validation",
    "runtime_validation_disabled",
]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ is not sorted")
<|endoftext|>import math
from collections.abc import Iterator
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


__all__ = ["DistributedSampler"]


_T_co = TypeVar("_T_co", covariant=True)


class DistributedSampler(Sampler[_T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        if len(indices) != self.total_size:
            raise AssertionError(
                f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
            )

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
            )

        # pyrefly: ignore [bad-return]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
<|endoftext|># mypy: allow-untyped-defs
import bisect
import itertools
import math
import warnings
from collections.abc import Sequence

# UP006 wants 'Iterable' to be imported from collections.abc but it needs to
# stay from typing for now due to BC concerns. In particular several internal
# targets fail to typecheck with:
#     TypeError: Cannot create a consistent method resolution order (MRO) for
#     bases Iterable, Generic
from typing import cast, Generic, Iterable, TypeVar  # noqa: UP035
from typing_extensions import deprecated

# No 'default_generator' in torch/__init__.pyi
from torch import default_generator, Generator, randperm, Tensor


__all__ = [
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "StackDataset",
    "ConcatDataset",
    "ChainDataset",
    "Subset",
    "random_split",
]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_dict = dict[str, _T_co]
_T_tuple = tuple[_T_co, ...]
_T_stack = TypeVar("_T_stack", _T_tuple, _T_dict)


class Dataset(Generic[_T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> _T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    # def __getitems__(self, indices: List) -> List[_T_co]:
    # Not implemented to prevent false-positives in fetcher check in
    # torch.utils.data._utils.fetch._MapDatasetFetcher

    def __add__(self, other: "Dataset[_T_co]") -> "ConcatDataset[_T_co]":
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py


class IterableDataset(Dataset[_T_co], Iterable[_T_co]):
    r"""An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this dataset.

    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the dataset object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

    Example 1: splitting workload across all workers in :meth:`__iter__`::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DATALOADER)
        >>> # xdoctest: +SKIP("Fails on MacOS12")
        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         worker_info = torch.utils.data.get_worker_info()
        ...         if worker_info is None:  # single-process data loading, return the full iterator
        ...             iter_start = self.start
        ...             iter_end = self.end
        ...         else:  # in a worker process
        ...             # split workload
        ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        ...             worker_id = worker_info.id
        ...             iter_start = self.start + worker_id * per_worker
        ...             iter_end = min(iter_start + per_worker, self.end)
        ...         return iter(range(iter_start, iter_end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [tensor([3]), tensor([4]), tensor([5]), tensor([6])]

        >>> # xdoctest: +REQUIRES(POSIX)
        >>> # Multi-process loading with two worker processes
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> # xdoctest: +IGNORE_WANT("non deterministic")
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [tensor([3]), tensor([5]), tensor([4]), tensor([6])]

        >>> # With even more workers
        >>> # xdoctest: +IGNORE_WANT("non deterministic")
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
        [tensor([3]), tensor([5]), tensor([4]), tensor([6])]

    Example 2: splitting workload across all workers using :attr:`worker_init_fn`::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DATALOADER)
        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [3, 4, 5, 6]
        >>>
        >>> # Directly doing multi-process loading yields duplicate data
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [3, 3, 4, 4, 5, 5, 6, 6]

        >>> # Define a `worker_init_fn` that configures each dataset copy differently
        >>> def worker_init_fn(worker_id):
        ...     worker_info = torch.utils.data.get_worker_info()
        ...     dataset = worker_info.dataset  # the dataset copy in this worker process
        ...     overall_start = dataset.start
        ...     overall_end = dataset.end
        ...     # configure the dataset to only process the split workload
        ...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        ...     worker_id = worker_info.id
        ...     dataset.start = overall_start + worker_id * per_worker
        ...     dataset.end = min(dataset.start + per_worker, overall_end)
        ...

        >>> # Multi-process loading with the custom `worker_init_fn`
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
        [3, 5, 4, 6]

        >>> # With even more workers
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=12, worker_init_fn=worker_init_fn)))
        [3, 4, 5, 6]
    """

    def __add__(self, other: Dataset[_T_co]):
        return ChainDataset([self, other])

    # No `def __len__(self)` default? Subclasses raise `TypeError` when needed.
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]


class TensorDataset(Dataset[tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    tensors: tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        if any(tensors[0].size(0) != tensor.size(0) for tensor in tensors):
            raise AssertionError("Size mismatch between tensors")
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        return self.tensors[0].size(0)


class StackDataset(Dataset[_T_stack]):
    r"""Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {"image": images[0], "text": texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """

    datasets: tuple | dict

    def __init__(self, *args: Dataset[_T_co], **kwargs: Dataset[_T_co]) -> None:
        if args:
            if kwargs:
                raise ValueError(
                    "Supported either ``tuple``- (via ``args``) or"
                    "``dict``- (via ``kwargs``) like input/output, but both types are given."
                )
            self._length = len(args[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in args):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = args
        elif kwargs:
            tmp = list(kwargs.values())
            self._length = len(tmp[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in tmp):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = kwargs
        else:
            raise ValueError("At least one dataset should be passed")

    def __getitem__(self, index):
        if isinstance(self.datasets, dict):
            return {k: dataset[index] for k, dataset in self.datasets.items()}
        return tuple(dataset[index] for dataset in self.datasets)

    def __getitems__(self, indices: list):
        # add batched sampling support when parent datasets supports it.
        if isinstance(self.datasets, dict):
            dict_batch: list[_T_dict] = [{} for _ in indices]
            for k, dataset in self.datasets.items():
                if callable(getattr(dataset, "__getitems__", None)):
                    items = dataset.__getitems__(indices)  # type: ignore[attr-defined]
                    if len(items) != len(indices):
                        raise ValueError(
                            "Nested dataset's output size mismatch."
                            f" Expected {len(indices)}, got {len(items)}"
                        )
                    for data, d_sample in zip(items, dict_batch, strict=True):
                        d_sample[k] = data
                else:
                    for idx, d_sample in zip(indices, dict_batch, strict=True):
                        d_sample[k] = dataset[idx]
            return dict_batch

        # tuple data
        list_batch: list[list] = [[] for _ in indices]
        for dataset in self.datasets:
            if callable(getattr(dataset, "__getitems__", None)):
                items = dataset.__getitems__(indices)  # type: ignore[attr-defined]
                if len(items) != len(indices):
                    raise ValueError(
                        "Nested dataset's output size mismatch."
                        f" Expected {len(indices)}, got {len(items)}"
                    )
                for data, t_sample in zip(items, list_batch, strict=True):
                    t_sample.append(data)
            else:
                for idx, t_sample in zip(indices, list_batch, strict=True):
                    t_sample.append(dataset[idx])
        tuple_batch: list[_T_tuple] = [tuple(sample) for sample in list_batch]
        return tuple_batch

    def __len__(self) -> int:
        return self._length


class ConcatDataset(Dataset[_T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: list[Dataset[_T_co]]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        if len(self.datasets) == 0:
            raise AssertionError("datasets should not be an empty iterable")
        for d in self.datasets:
            if isinstance(d, IterableDataset):
                raise AssertionError("ConcatDataset does not support IterableDataset")
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    @deprecated(
        "`cummulative_sizes` attribute is renamed to `cumulative_sizes`",
        category=FutureWarning,
    )
    def cummulative_sizes(self):
        return self.cumulative_sizes


class ChainDataset(IterableDataset):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            if not isinstance(d, IterableDataset):
                raise AssertionError("ChainDataset only supports IterableDataset")
            yield from d

    def __len__(self) -> int:
        total = 0
        for d in self.datasets:
            if not isinstance(d, IterableDataset):
                raise AssertionError("ChainDataset only supports IterableDataset")
            total += len(d)  # type: ignore[arg-type]
        return total


class Subset(Dataset[_T_co]):
    r"""
    Subset of a dataset at specified indices.

    .. note::
        When subclassing `Subset` and overriding `__getitem__`, you **must** also
        override `__getitems__` to ensure `DataLoader` works correctly with your
        custom logic. If you override only `__getitem__`, a `NotImplementedError`
        will be raised when using `DataLoader`.

        A simple implementation of `__getitems__` can delegate to `__getitem__`:

        .. code-block:: python

            def __getitems__(self, indices):
                return [self.__getitem__(idx) for idx in indices]

        For better performance, consider implementing batch-aware logic in
        `__getitems__` instead of calling `__getitem__` multiple times.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[_T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

        # Check if __getitem__ is overridden but __getitems__ is not
        if (
            type(self).__getitem__ is not Subset.__getitem__
            and type(self).__getitems__ is Subset.__getitems__
        ):
            raise NotImplementedError(
                f"{type(self).__name__} overrides __getitem__ but not __getitems__. "
                "When subclassing Subset and overriding __getitem__, you must also override "
                "__getitems__ to ensure DataLoader works correctly with your custom logic. "
                "A simple implementation:\n\n"
                "def __getitems__(self, indices):\n"
                "    return [self.__getitem__(idx) for idx in indices]"
            )

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: list[int]) -> list[_T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)


def random_split(
    dataset: Dataset[_T],
    lengths: Sequence[int | float],
    generator: Generator | None = default_generator,
) -> list[Subset[_T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset.",
                    stacklevel=2,
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths, strict=True)
    ]
<|endoftext|># mypy: allow-untyped-defs
r"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter.

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import logging
import multiprocessing as python_multiprocessing
import os
import queue
import threading
import warnings
import weakref
from collections.abc import Callable
from typing import Any, Generic, NoReturn, TYPE_CHECKING, TypeVar
from typing_extensions import Self

import torch
import torch.distributed as dist
import torch.utils.data.graph_settings
from torch._utils import ExceptionWrapper
from torch.utils.data import _utils
from torch.utils.data.datapipes.datapipe import (
    _IterDataPipeSerializationWrapper,
    _MapDataPipeSerializationWrapper,
    IterDataPipe,
    MapDataPipe,
)
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "DataLoader",
    "get_worker_info",
    "default_collate",
    "default_convert",
]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_worker_init_fn_t = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`, but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[list[_T]], Any]


# These functions used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate: _collate_fn_t = _utils.collate.default_collate
default_convert = _utils.collate.default_convert

get_worker_info = _utils.worker.get_worker_info

logger = logging.getLogger(__name__)

_persistent_workers_atexit_lock = threading.Lock()
_persistent_workers_atexit_registered = False
_persistent_workers_atexit: weakref.WeakSet[_MultiProcessingDataLoaderIter] = (
    weakref.WeakSet()
)


class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )
        else:
            return _utils.fetch._IterableDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )


class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.

    Used as sampler for :class:`~torch.utils.data.IterableDataset`.
    """

    def __iter__(self):
        while True:
            yield None


def _get_distributed_settings():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id) -> None:
    global_worker_id = worker_id
    info = torch.utils.data.get_worker_info()
    if info is None:
        raise AssertionError("Worker info is None in sharding worker init function")
    total_workers = info.num_workers
    datapipe = info.dataset
    if not isinstance(datapipe, (IterDataPipe, MapDataPipe)):
        raise AssertionError(
            "datapipe must be an instance of IterDataPipe or MapDataPipe"
        )
    # To distribute elements across distributed process evenly, we should shard data on distributed
    # processes first then shard on worker processes
    total_workers *= world_size
    global_worker_id = global_worker_id * world_size + rank_id
    # For BC, use default SHARDING_PRIORITIES
    torch.utils.data.graph_settings.apply_sharding(
        datapipe, total_workers, global_worker_id
    )
    if worker_init_fn is not None:
        worker_init_fn(worker_id)


def _share_dist_seed(generator, pg):
    _shared_seed = torch.empty((), dtype=torch.int64).random_(generator=generator)
    if isinstance(pg, dist.ProcessGroup):
        dist.broadcast(_shared_seed, src=0, group=pg)
    return _shared_seed.item()


class DataLoader(Generic[_T_co]):
    r"""
    Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (Callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (Callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        multiprocessing_context (str or multiprocessing.context.BaseContext, optional): If
            ``None``, the default
            `multiprocessing context <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_ # noqa: D401
            of your operating system will
            be used. (default: ``None``)
        generator (torch.Generator, optional): If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            ``base_seed`` for workers. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers batches prefetched across all workers. (default value depends
            on the set value for num_workers. If value of num_workers=0 default is ``None``.
            Otherwise, if value of ``num_workers > 0`` default is ``2``).
        persistent_workers (bool, optional): If ``True``, the data loader will not shut down
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
        pin_memory_device (str, optional): Deprecated, the current :ref:`accelerator<accelerators>`
            will be used as the device if ``pin_memory=True``.
        in_order (bool, optional): If ``False``, the data loader will not enforce that batches
            are returned in a first-in, first-out order. Only applies when ``num_workers > 0``. (default: ``True``)


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.

    .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                 :ref:`data-loading-randomness` notes for random seed related questions.

    .. warning:: Setting `in_order` to `False` can harm reproducibility and may lead to a skewed data
                 distribution being fed to the trainer in cases with imbalanced data.
    """

    dataset: Dataset[_T_co]
    batch_size: int | None
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Sampler | Iterable
    pin_memory_device: str
    prefetch_factor: int | None
    _iterator: _BaseDataLoaderIter | None
    __initialized = False

    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[list] | Iterable[list] | None = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: _worker_init_fn_t | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> None:
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; "
                "use num_workers=0 to disable multiprocessing."
            )

        if timeout < 0:
            raise ValueError("timeout option should be non-negative")

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing."
                "let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None."
            )
        elif num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError("prefetch_factor option should be non-negative")

        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs num_workers > 0")

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.in_order = in_order

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   _DataPipeSerializationWrapper container makes it easier to serialize without redefining pickler
        if isinstance(self.dataset, IterDataPipe):
            self.dataset = _IterDataPipeSerializationWrapper(self.dataset)
        elif isinstance(self.dataset, MapDataPipe):
            self.dataset = _MapDataPipeSerializationWrapper(self.dataset)

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and IterableDataset ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data loss (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `IterableDataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if isinstance(dataset, IterDataPipe):
                if shuffle is not None:
                    dataset = torch.utils.data.graph_settings.apply_shuffle_settings(
                        dataset, shuffle=shuffle
                    )
            # We cannot check `shuffle is not None` here, since previously `shuffle=False` was the default.
            elif shuffle not in {False, None}:
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle={shuffle}"
                )

            if sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified sampler option, but got sampler={sampler}"
                )
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    f"batch_sampler option, but got batch_sampler={batch_sampler}"
                )
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError(
                    "batch_size=None option disables auto-batching "
                    "and is mutually exclusive with drop_last"
                )

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                else:
                    sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = (
            None  # See NOTE [ IterableDataset and __len__ ]
        )

        self._iterator = None

        self.check_worker_number_rationality()

    def _get_iterator(self) -> _BaseDataLoaderIter:
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context) -> None:
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = torch.multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            "multiprocessing_context option "
                            f"should specify a valid start method in {valid_start_methods!r}, but got "
                            f"multiprocessing_context={multiprocessing_context!r}"
                        )
                    multiprocessing_context = torch.multiprocessing.get_context(
                        multiprocessing_context
                    )

                if not isinstance(
                    multiprocessing_context, python_multiprocessing.context.BaseContext
                ):
                    raise TypeError(
                        "multiprocessing_context option should be a valid context "
                        "object or a string specifying the start method, but got "
                        f"multiprocessing_context={multiprocessing_context}"
                    )
            else:
                raise ValueError(
                    "multiprocessing_context can only be used with "
                    "multi-process loading (num_workers > 0), but got "
                    f"num_workers={self.num_workers}"
                )

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val) -> None:
        if self.__initialized and attr in (
            "batch_size",
            "batch_sampler",
            "sampler",
            "drop_last",
            "dataset",
            "persistent_workers",
        ):
            raise ValueError(
                f"{attr} attribute should not be set after {self.__class__.__name__} is initialized"
            )

        super().__setattr__(attr, val)

    def __iter__(self) -> _BaseDataLoaderIter:
        # When using a single worker the returned iterator should be
        # created every time to avoid resetting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            # NOTE [ IterableDataset and __len__ ]
            #
            # For `IterableDataset`, `__len__` could be inaccurate when one naively
            # does multi-processing data loading, since the samples will be duplicated.
            # However, no real use case should be actually using that behavior, so
            # it should count as a user error. We should generally trust user
            # code to do the proper thing (e.g., configure each replica differently
            # in `__iter__`), and give us the correct `__len__` if they choose to
            # implement it (this will still throw if the dataset does not implement
            # a `__len__`).
            #
            # To provide a further warning, we track if `__len__` was called on the
            # `DataLoader`, save the returned value in `self._len_called`, and warn
            # if the iterator ends up yielding more than this number of samples.

            # Cannot statically verify that dataset is Sized
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore[assignment, arg-type]
            if (
                self.batch_size is not None
            ):  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil

                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)

    def check_worker_number_rationality(self) -> None:
        # This function check whether the dataloader's worker number is rational based on
        # current system's resource. Current rule is that if the number of workers this
        # Dataloader will create is bigger than the number of logical cpus that is allowed to
        # use, then we will pop up a warning to let user pay attention.
        #
        # eg. If current system has 2 physical CPUs with 16 cores each. And each core support 2
        #     threads, then the total logical cpus here is 2 * 16 * 2 = 64. Let's say current
        #     DataLoader process can use half of them which is 32, then the rational max number of
        #     worker that initiated from this process is 32.
        #     Now, let's say the created DataLoader has num_works = 40, which is bigger than 32.
        #     So the warning message is triggered to notify the user to lower the worker number if
        #     necessary.
        #
        #
        # [Note] Please note that this function respects `cpuset` only when os.sched_getaffinity is
        #        available (available in most of Linux system, but not OSX and Windows).
        #        When os.sched_getaffinity is not available, os.cpu_count() is called instead, but
        #        it doesn't respect cpuset.
        #        We don't take threading into account since each worker process is single threaded
        #        at this time.
        #
        #        We don't set any threading flags (eg. OMP_NUM_THREADS, MKL_NUM_THREADS, etc)
        #        other than `torch.set_num_threads` to 1 in the worker process, if the passing
        #        in functions use 3rd party modules that rely on those threading flags to determine
        #        how many thread to create (eg. numpy, etc), then it is caller's responsibility to
        #        set those flags correctly.
        def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):
            suggested_max_worker_msg = (
                (
                    (
                        "Our suggested max number of worker in current system is {}{}, which is smaller "
                        "than what this DataLoader is going to create."
                    ).format(
                        num_worker_suggest,
                        (
                            ""
                            if cpuset_checked
                            else " (`cpuset` is not taken into account)"
                        ),
                    )
                )
                if num_worker_suggest is not None
                else (
                    "DataLoader is not able to compute a suggested max number of worker in current system."
                )
            )

            warn_msg = (
                f"This DataLoader will create {num_worker_created} worker processes in total. {suggested_max_worker_msg} "
                "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
                "lower the worker number to avoid potential slowness/freeze if necessary."
            )
            return warn_msg

        if not self.num_workers or self.num_workers == 0:
            return

        # try to compute a suggested max number of worker based on system's resource
        max_num_worker_suggest = torch._utils.cpu_count()
        cpuset_checked = hasattr(os, "sched_getaffinity")

        if max_num_worker_suggest is None or self.num_workers > max_num_worker_suggest:
            warnings.warn(
                _create_warning_msg(
                    max_num_worker_suggest, self.num_workers, cpuset_checked
                ),
                stacklevel=2,
            )


class _BaseDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._shared_seed = None
        self._pg = None
        if isinstance(self._dataset, IterDataPipe):
            if dist.is_available() and dist.is_initialized():
                self._pg = dist.new_group(backend="gloo")
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(
                self._dataset, shared_rng
            )
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        ws, rank = _get_distributed_settings()
        self._world_size = ws
        self._rank = rank

        if loader.pin_memory and loader.pin_memory_device:
            warnings.warn(
                "pin_memory_device is deprecated, the current accelerator will be used as the device,"
                f"ignore pin_memory_device='{loader.pin_memory_device}'.",
                stacklevel=2,
            )
        if loader.pin_memory and not torch.accelerator.is_available():
            warn_msg = (
                "'pin_memory' argument is set as true but no accelerator is found, "
                "then device pinned memory won't be used."
            )
            warnings.warn(warn_msg, stacklevel=2)

        # Enabling pin_memory in _BaseDataLoaderIter to support identical
        # behavior in forked implementations using _BaseDataLoaderIter.
        self._pin_memory = loader.pin_memory and torch.accelerator.is_available()

        # Set pin memory device based on the current accelerator.
        self._pin_memory_device = (
            acc.type
            if self._pin_memory
            and (acc := torch.accelerator.current_accelerator()) is not None
            else None
        )

        # Currently, pin_memory would raise error on the MPS backend (see
        # https://github.com/pytorch/pytorch/issues/86060), so forcibly
        # disable pin_memory on MPS. Remove this restriction once pinned
        # memory allocation for MPS is fixed.
        if self._pin_memory_device == "mps":
            self._pin_memory = False
            warn_msg = (
                "'pin_memory' argument is set as true but not supported on MPS now, "
                "device pinned memory won't be used."
            )
            warnings.warn(warn_msg, stacklevel=2)

        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = (
            torch.empty((), dtype=torch.int64)
            .random_(generator=loader.generator)
            .item()
        )
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = f"enumerate(DataLoader)#{self.__class__.__name__}.__next__"

    def __iter__(self) -> Self:
        return self

    def _reset(self, loader, first_iter=False) -> None:
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        if isinstance(self._dataset, IterDataPipe):
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(
                self._dataset, shared_rng
            )

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self) -> NoReturn:
        raise NotImplementedError

    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                # TODO(https://github.com/pytorch/pytorch/issues/76750)
                self._reset()  # type: ignore[call-arg]
            data = self._next_data()
            self._num_yielded += 1
            if (
                self._dataset_kind == _DatasetKind.Iterable
                and self._IterableDataset_len_called is not None
                and self._num_yielded > self._IterableDataset_len_called
            ):
                warn_msg = (
                    f"Length of IterableDataset {self._dataset} was reported to be {self._IterableDataset_len_called}"
                    f"(when accessing len(dataloader)), but {self._num_yielded} samples have been fetched. "
                )
                if self._num_workers > 0:
                    warn_msg += (
                        "For multiprocessing data-loading, this could be caused by not properly configuring the "
                        "IterableDataset replica at each worker. Please see "
                        "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples."
                    )
                warnings.warn(warn_msg, stacklevel=2)
            return data

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader) -> None:
        super().__init__(loader)
        if self._timeout != 0:
            raise AssertionError("_SingleProcessDataLoaderIter requires timeout == 0")
        if self._num_workers != 0:
            raise AssertionError(
                "_SingleProcessDataLoaderIter requires num_workers == 0"
            )

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Taking care of distributed sharding
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            # For BC, use default SHARDING_PRIORITIES
            torch.utils.data.graph_settings.apply_sharding(
                self._dataset, self._world_size, self._rank
            )

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler."""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed (e.g., releasing GPU memory).
    #      Naturally, we implement the shutdown logic in `__del__` of
    #      DataLoaderIterator.
    #
    #      We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers and `pin_memory_thread` to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when the
    #      iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may already be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children.
    #
    #      Thus, we register an `atexit` hook that sets a global flag
    #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
    #      reverse order of registration, we are guaranteed that this flag is
    #      set before library resources we use are freed (which, at least in
    #      CPython, is done via an `atexit` handler defined in
    #      `multiprocessing/util.py`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/util.py#L320-L362
    #      registered when an object requiring this mechanism is first
    #      created, e.g., `mp.Queue`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/context.py#L100-L103
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/queues.py#L29
    #      )
    #
    #      So in `__del__`, we check if `_utils.python_exit_status` is set or
    #      `None` (freed), and perform no-op if so.
    #
    #      However, simply letting library clean-up codes run can also be bad,
    #      because such codes (i.e., `multiprocessing.util._exit_function()`)
    #      include join putting threads for `mp.Queue`, which can be blocking.
    #      Hence, the main process putting threads are called with
    #      `cancel_join_thread` at creation.  See later section
    #      [ 3b. A process won't hang when putting into a queue; ]
    #      for more details.
    #
    #      Here are two example cases where library clean-up codes can run
    #      before `__del__` is called:
    #
    #        1. If we hold onto a reference to the iterator, it more often
    #           than not tries to do `multiprocessing` library cleaning before
    #           clearing the alive referenced objects (https://github.com/pytorch/pytorch/issues/48666)
    #           and thus prevents our cleaning-up code to run first.
    #
    #        2. A similar issue araises when a `DataLoader` is used in a subprocess.
    #           When a process ends, it shuts the all its daemonic children
    #           down with a SIGTERM (instead of joining them without a timeout).
    #           Similarly for threads, but by a different mechanism. This fact,
    #           together with a few implementation details of multiprocessing, forces
    #           us to make workers daemonic. All of our problems arise when a
    #           DataLoader is used in a subprocess, and are caused by multiprocessing
    #           code which looks more or less like this:
    #
    #               try:
    #                   your_function_using_a_dataloader()
    #               finally:
    #                   multiprocessing.util._exit_function()
    #
    #           The joining/termination mentioned above happens inside
    #           `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #           throws, the stack trace stored in the exception will prevent the
    #           frame which uses `DataLoaderIter` to be freed. If the frame has any
    #           reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #           its  `__del__`, which starts the shutdown procedure, will not be
    #           called. That, in turn, means that workers aren't notified. Attempting
    #           to join in `_exit_function` will then result in a hang.
    #
    #           For context, `_exit_function` is also registered as an `atexit` call.
    #           So it is unclear to me (@ssnl) why this is needed in a finally block.
    #           The code dates back to 2008 and there is no comment on the original
    #           PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #           the finally block and the `atexit` registration) that explains this.
    #
    #
    #      Finally, another choice is to just shutdown workers with logic in 1
    #      above whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #           always corresponding to a `get()`), hanging on `get()` can still
    #           happen when data in queue is corrupted (e.g., due to
    #           `cancel_join_thread` or unexpected exit).
    #
    #           For child exit, we set a timeout whenever we try to get data
    #           from `data_queue`, and check the workers' status on each timeout
    #           and error.
    #           See `_DataLoaderiter._get_batch()` and
    #           `_DataLoaderiter._try_get_data()` for details.
    #
    #           Additionally, for child exit on non-Windows platforms, we also
    #           register a SIGCHLD handler (which is supported on Windows) on
    #           the main process, which checks if any of the workers fail in the
    #           (Python) handler. This is more efficient and faster in detecting
    #           worker failures, compared to only using the above mechanism.
    #           See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
    #
    #           For `.get()` calls where the sender(s) is not the workers, we
    #           guard them with timeouts, and check the status of the sender
    #           when timeout happens:
    #             + in the workers, the `_utils.worker.ManagerWatchdog` class
    #               checks the status of the main process.
    #             + if `pin_memory=True`, when getting from `pin_memory_thread`,
    #               check `pin_memory_thread` status periodically until `.get()`
    #               returns or see that `pin_memory_thread` died.
    #
    #        b. A process won't hang when putting into a queue;
    #
    #           We use `mp.Queue` which has a separate background thread to put
    #           objects from an unbounded buffer array. The background thread is
    #           daemonic and usually automatically joined when the process
    #           *exits*.
    #
    #           In case that the receiver has ended abruptly while
    #           reading from the pipe, the join will hang forever.  The usual
    #           solution for this in Python is calling  `q.cancel_join_thread`,
    #           which prevents automatically joining it when finalizing
    #           (exiting).
    #
    #           Nonetheless, `cancel_join_thread` must only be called when the
    #           queue is **not** going to be read from or write into by another
    #           process, because it may hold onto a lock or leave corrupted data
    #           in the queue, leading other readers/writers to hang.
    #
    #           Hence,
    #             + For worker processes, we only do so (for their output
    #               queues, i.e., `worker_result_queue`) before exiting.
    #             + For `pin_memory_thread`, its output queue `data_queue` is a
    #               `queue.Queue` that does blocking `put` if the queue is full.
    #               So there is no above problem, but as a result, in
    #               `_pin_memory_loop`, we do need to  wrap the `put` in a loop
    #               that breaks not only upon success, but also when the main
    #               process stops reading, i.e., is shutting down.
    #             + For loader process, we `cancel_join_thread()` for all
    #               `_index_queues` because the whole purpose of workers and
    #               `pin_memory_thread` is to serve the loader process.  If
    #               loader process is already exiting, we don't really care if
    #               the queues are corrupted.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # `workers_done_event`:
    #   A `multiprocessing.Event` shared among the main process and all worker
    #   processes. This is used to signal the workers that the iterator is
    #   shutting down. After it is set, they will not send processed data to
    #   queues anymore, and only wait for the final `None` before exiting.
    #   `done_event` isn't strictly needed. I.e., we can just check for `None`
    #   from the input queue, but it allows us to skip wasting resources
    #   processing data if we are already shutting down.
    #
    # `pin_memory_thread_done_event`:
    #   A `threading.Event` for a similar purpose to that of
    #   `workers_done_event`, but is for the `pin_memory_thread`. The reason
    #   that separate events are needed is that `pin_memory_thread` reads from
    #   the output queue of the workers. But the workers, upon seeing that
    #   `workers_done_event` is set, only wants to see the final `None`, and is
    #   not required to flush all data in the output queue (e.g., it may call
    #   `cancel_join_thread` on that queue if its `IterableDataset` iterator
    #   happens to exhaust coincidentally, which is out of the control of the
    #   main process). Thus, since we will exit `pin_memory_thread` before the
    #   workers (see below), two separate events are used.
    #
    # NOTE: In short, the protocol is that the main process will set these
    #       `done_event`s and then the corresponding processes/threads a `None`,
    #       and that they may exit at any time after receiving the `None`.
    #
    # NOTE: Using `None` as the final signal is valid, since normal data will
    #       always be a 2-tuple with the 1st element being the index of the data
    #       transferred (different from dataset index/key), and the 2nd being
    #       either the dataset key or the data sample (depending on which part
    #       of the data model the queue is at).
    #
    # [ worker processes ]
    #   While loader process is alive:
    #     Get from `index_queue`.
    #       If get anything else,
    #          Check `workers_done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data:
    #                If is fetching from an `IterableDataset` and the iterator
    #                    is exhausted, send an `_IterableDatasetStopIteration`
    #                    object to signal iteration end. The main process, upon
    #                    receiving such an object, will send `None` to this
    #                    worker and not use the corresponding `index_queue`
    #                    anymore.
    #       If timed out,
    #          No matter `workers_done_event` is set (still need to see `None`)
    #          or not, must continue to next iteration.
    #   (outside loop)
    #   If `workers_done_event` is set,  (this can be False with `IterableDataset`)
    #     `data_queue.cancel_join_thread()`.  (Everything is ending here:
    #                                          main process won't read from it;
    #                                          other workers will also call
    #                                          `cancel_join_thread`.)
    #
    # [ pin_memory_thread ]
    #   # No need to check main thread. If this thread is alive, the main loader
    #   # thread must be alive, because this thread is set as daemonic.
    #   While `pin_memory_thread_done_event` is not set:
    #     Get from `worker_result_queue`.
    #       If timed out, continue to get in the next iteration.
    #       Otherwise, process data.
    #       While `pin_memory_thread_done_event` is not set:
    #         Put processed data to `data_queue` (a `queue.Queue` with blocking put)
    #         If timed out, continue to put in the next iteration.
    #         Otherwise, break, i.e., continuing to the out loop.
    #
    #   NOTE: we don't check the status of the main thread because
    #           1. if the process is killed by fatal signal, `pin_memory_thread`
    #              ends.
    #           2. in other cases, either the cleaning-up in __del__ or the
    #              automatic exit of daemonic thread will take care of it.
    #              This won't busy-wait either because `.get(timeout)` does not
    #              busy-wait.
    #
    # [ main process ]
    #   In the DataLoader Iter's `__del__`
    #     b. Exit `pin_memory_thread`
    #          i.   Set `pin_memory_thread_done_event`.
    #          ii   Put `None` in `worker_result_queue`.
    #          iii. Join the `pin_memory_thread`.
    #          iv.  `worker_result_queue.cancel_join_thread()`.
    #
    #     c. Exit the workers.
    #          i.   Set `workers_done_event`.
    #          ii.  Put `None` in each worker's `index_queue`.
    #          iii. Join the workers.
    #          iv.  Call `.cancel_join_thread()` on each worker's `index_queue`.
    #
    #        NOTE: (c) is better placed after (b) because it may leave corrupted
    #              data in `worker_result_queue`, which `pin_memory_thread`
    #              reads from, in which case the `pin_memory_thread` can only
    #              happen at timing out, which is slow. Nonetheless, same thing
    #              happens if a worker is killed by signal at unfortunate times,
    #              but in other cases, we are better off having a non-corrupted
    #              `worker_result_queue` for `pin_memory_thread`.
    #
    #   NOTE: If `pin_memory=False`, there is no `pin_memory_thread` and (b)
    #         can be omitted
    #
    # NB: `done_event`s isn't strictly needed. E.g., we can just check for
    #     `None` from `index_queue`, but it allows us to skip wasting resources
    #     processing indices already in `index_queue` if we are already shutting
    #     down.

    def __init__(self, loader) -> None:
        super().__init__(loader)

        self._prefetch_factor = loader.prefetch_factor
        self._in_order = loader.in_order

        if self._num_workers <= 0:
            raise AssertionError(
                "num_workers must be greater than 0 for MultiProcessingDataLoaderIter"
            )
        if self._prefetch_factor <= 0:
            raise AssertionError(
                "prefetch_factor must be greater than 0 for MultiProcessingDataLoaderIter"
            )

        if loader.multiprocessing_context is None:
            multiprocessing_context = torch.multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Additional worker init function will take care of sharding in MP and Distributed
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            self._worker_init_fn = functools.partial(
                _sharding_worker_init_fn,
                self._worker_init_fn,
                self._world_size,
                self._rank,
            )

        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            # The worker consumes the holder so its Process does not retain the dataset.
            dataset_holder = [self._dataset]
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(
                    self._dataset_kind,
                    dataset_holder,
                    index_queue,
                    self._worker_result_queue,
                    self._workers_done_event,
                    self._auto_collation,
                    self._collate_fn,
                    self._drop_last,
                    self._base_seed,
                    self._worker_init_fn,
                    i,
                    self._num_workers,
                    self._persistent_workers,
                    self._shared_seed,
                ),
            )
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            from pickle import PicklingError

            try:
                w.start()
            except (TypeError, AttributeError, PicklingError):
                warnings.warn(
                    "Got pickle error when attempting to start a worker Process. "
                    "This might be because the worker Process arguments are not picklable. "
                    "Python 3.14+ changed the multiprocessing start method in non-Mac POSIX platforms "
                    "to 'forkserver', which requires the worker Process arguments to be picklable. "
                    "You can also try multiprocessing.set_start_method('fork').",
                    stacklevel=2,
                )
                raise
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            current_device_id = torch.accelerator.current_device_index()
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(
                    self._worker_result_queue,
                    self._data_queue,
                    current_device_id,
                    self._pin_memory_thread_done_event,
                    self._pin_memory_device,
                ),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue  # type: ignore[assignment]

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            self._register_persistent_workers_atexit()

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(
            id(self),
            tuple(w.pid for w in self._workers),  # type: ignore[misc]
        )
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False) -> None:
        super()._reset(loader, first_iter)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = (
            0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        )
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        # Not that this indicates that a worker still has work to do *for this epoch*.
        # It does not mean that a worker is dead. In case of `_persistent_workers`,
        # the worker will be reset to available in the next epoch.
        self._workers_status = [True for i in range(self._num_workers)]
        # A list of integers representing how many tasks are outstanding for each worker
        # Incremented when a task is dispatched to the worker
        # Decremented when that data has been given to the main thread
        # Each worker should have at most self._prefetch_factor tasks outstanding
        self._workers_num_tasks = [0 for i in range(self._num_workers)]
        # Reset the worker queue cycle so it resumes next epoch at worker 0
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(
                    _utils.worker._ResumeIteration(self._shared_seed)
                )
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _utils.worker._ResumeIteration):
                    if return_data is not None:
                        raise AssertionError(
                            "Expected return_data to be None when resuming iteration"
                        )
                    resume_iteration_cnt -= 1
        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ", ".join(str(w.pid) for w in failed_workers)
                raise RuntimeError(
                    f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
                ) from e
            if isinstance(e, queue.Empty):
                return (False, None)

            import errno
            import tempfile

            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                with contextlib.ExitStack() as stack:
                    for _ in range(fds_limit_margin):
                        stack.enter_context(
                            tempfile.NamedTemporaryFile()  # pyrefly: ignore [bad-argument-type]
                        )
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code"
                    ) from None
            raise

    # NOTE [ DataLoader on Linux and open files limit ]
    #
    # On Linux when DataLoader is used with multiprocessing we pass the data between
    # the root process and the workers through SHM files. We remove those files from
    # the filesystem as soon as they are created and keep them alive by
    # passing around their file descriptors through AF_UNIX sockets. (See
    # docs/source/multiprocessing.rst and 'Multiprocessing Technical Notes` in
    # the wiki (https://github.com/pytorch/pytorch/wiki).)
    #
    # This sometimes leads us to exceeding the open files limit. When that happens,
    # and the offending file descriptor is coming over a socket, the `socket` Python
    # package silently strips the file descriptor from the message, setting only the
    # `MSG_CTRUNC` flag (which might be a bit misleading since the manpage says that
    # it _indicates that some control data were discarded due to lack of space in
    # the buffer for ancillary data_). This might reflect the C implementation of
    # AF_UNIX sockets.
    #
    # This behaviour can be reproduced with the script and instructions at the
    # bottom of this note.
    #
    # When that happens, the standard Python `multiprocessing` (and not
    # `torch.multiprocessing`) raises a `RuntimeError: received 0 items of ancdata`
    #
    # Sometimes, instead of the FD being stripped, you may get an `OSError:
    # Too many open files`, both in the script below and in DataLoader. However,
    # this is rare and seems to be nondeterministic.
    #
    #
    #   #!/usr/bin/env python3
    #   import sys
    #   import socket
    #   import os
    #   import array
    #   import shutil
    #   import socket
    #
    #
    #   if len(sys.argv) != 4:
    #       print("Usage: ", sys.argv[0], " tmp_dirname iteration (send|recv)")
    #       sys.exit(1)
    #
    #   if __name__ == '__main__':
    #       dirname = sys.argv[1]
    #       sock_path = dirname + "/sock"
    #       iterations = int(sys.argv[2])
    #       def dummy_path(i):
    #           return dirname + "/" + str(i) + ".dummy"
    #
    #
    #       if sys.argv[3] == 'send':
    #           while not os.path.exists(sock_path):
    #               pass
    #           client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    #           client.connect(sock_path)
    #           for i in range(iterations):
    #               fd = os.open(dummy_path(i), os.O_WRONLY | os.O_CREAT)
    #               ancdata = array.array('i', [fd])
    #               msg = bytes([i % 256])
    #               print("Sending fd ", fd, " (iteration #", i, ")")
    #               client.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, ancdata)])
    #
    #
    #       else:
    #           assert sys.argv[3] == 'recv'
    #
    #           if os.path.exists(dirname):
    #               raise Exception("Directory exists")
    #
    #           os.mkdir(dirname)
    #
    #           print("Opening socket...")
    #           server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    #           server.bind(sock_path)
    #
    #           print("Listening...")
    #           for i in range(iterations):
    #               a = array.array('i')
    #               msg, ancdata, flags, addr = server.recvmsg(1, socket.CMSG_SPACE(a.itemsize))
    #               assert(len(ancdata) == 1)
    #               cmsg_level, cmsg_type, cmsg_data = ancdata[0]
    #               a.frombytes(cmsg_data)
    #               print("Received fd ", a[0], " (iteration #", i, ")")
    #
    #           shutil.rmtree(dirname)
    #
    # Steps to reproduce:
    #
    # 1. Run two shells and set lower file descriptor limit in the receiving one:
    # (shell1) ulimit -n 1020
    # (shell2) ulimit -n 1022
    #
    # 2. Run the script above with the `recv` option in the first shell
    # (shell1) ./test_socket.py sock_tmp 1017 recv
    #
