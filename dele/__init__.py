#pylint: disable=unused-import

from pkg_resources import resource_string

__version__ = resource_string(__name__, "VERSION").decode("ascii").strip()

from . import utils
from . import dataset
from . import arch_network
from . import arch_embedder
