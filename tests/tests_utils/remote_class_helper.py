from rlightning.utils.ray.remote_class import RayActorMixin


class Custom(RayActorMixin):
    def __init__(self):
        super().__init__()
