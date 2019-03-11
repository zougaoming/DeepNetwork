#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .Graph import *
from .ActiveFunc import *
from .Gate import *
from .Json2Network import *
from .Loss import *
from .NetworkByJson import *
from .Optimizer import *
from .Save import *

# Create a default graph.
import builtins
DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()
