try:
    import zmq  # noqa: F401
    import streamz_zmq  # noqa: F401
    from streamz import Stream

    is_streaming_available = True

    @Stream.register_api()
    class epoch_window_via_dict(Stream):
        """
        Groups incoming elements into buffers that are emitted when an
        element with the attribute .name == 'epoch.end' is received.

        The emitted value is a tuple containing (epoch_number, list_of_elements).
        """

        _graphviz_shape = 'diamond'

        def __init__(self, upstream, **kwargs):
            self._buffer = []
            self.epoch_number = 0
            Stream.__init__(self, upstream, **kwargs)

        def update(self, x, who=None, metadata=None):
            # print('Received line', x)
            if x.get('name') == 'epoch.start':
                self.epoch_number += 1
                print('epoch start', self.epoch_number)

            if self.epoch_number > 0:
                x['epoch'] = self.epoch_number
                self._buffer.append(x)

            if x.get('name') == 'epoch.end':
                print('epoch end', self.epoch_number)
                data_to_emit = self._buffer
                self._buffer = []
                return self._emit(data_to_emit)

except ImportError:
    Stream = None

    is_streaming_available = False
