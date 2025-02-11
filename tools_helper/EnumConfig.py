# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 15:31
# @Author  : zhanghaoxiang
# @File    : EnumConfig.py
# @Software: PyCharm
import sys
from enum import *
class TypeEnum(Enum):
    chat: int = 1
    generate: int = 2
    create: int = 3