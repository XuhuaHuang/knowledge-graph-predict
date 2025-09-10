import inspect, torch
from torch_scatter import scatter


def scatter_(name, src, index, dim_size=None):
    if name == "add":
        name = "sum"
    assert name in ["sum", "mean", "max"]
    out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
    return out[0] if isinstance(out, tuple) else out


class MessagePassing(torch.nn.Module):
    def __init__(self, aggr="add"):
        # super(MessagePassing, self).__init__()
        super().__init__()

        # Get the underlying function object for bound methods (if any)
        # and use getfullargspec which works with function objects.
        msg_func = getattr(self.message, "__func__", self.message)
        upd_func = getattr(self.update, "__func__", self.update)

        # args: for message we skip the first arg (self), for update skip first two (self, out)
        self.message_args = inspect.getfullargspec(msg_func).args[1:]
        self.update_args = inspect.getfullargspec(upd_func).args[2:]

    def propagate(self, aggr, edge_index, **kwargs):
        assert aggr in ["add", "mean", "max"]
        kwargs["edge_index"] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == "_i":  # If arguments ends with _i then include indic
                tmp = kwargs[
                    arg[:-2]
                ]  # Take the front part of the variable | Mostly it will be 'x',
                size = tmp.size(0)
                message_args.append(
                    tmp[edge_index[0]]
                )  # Lookup for head entities in edges
            elif arg[-2:] == "_j":
                tmp = kwargs[arg[:-2]]  # tmp = kwargs['x']
                size = tmp.size(0)
                message_args.append(
                    tmp[edge_index[1]]
                )  # Lookup for tail entities in edges
            else:
                message_args.append(kwargs[arg])  # Take things from kwargs

        update_args = [
            kwargs[arg] for arg in self.update_args
        ]  # Take update args from kwargs

        out = self.message(*message_args)
        out = scatter_(
            aggr, out, edge_index[0], dim_size=size
        )  # Aggregated neighbors for each vertex
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def update(self, aggr_out):  # pragma: no cover
        return aggr_out
