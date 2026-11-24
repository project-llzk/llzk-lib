#!/usr/bin/env python3
# Part of the LLZK Project, under the Apache License v2.0.
# See LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0

"""Generate aggregate copies with many concrete leaves or distinct ranged fields."""
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('kind', choices=['concrete', 'ranged'])
parser.add_argument('count', type=int)
args = parser.parse_args()
n = args.count
if args.kind == 'concrete':
    array = f'!array.type<{n} x !felt.type>'
    print(f'!W = !pod.type<[@array: {array}]>')
    name = 'ConcreteCopyScale'
    arguments = f'%array: {array}'
else:
    print('!P = !pod.type<[' + ', '.join(f'@field{i}: !felt.type' for i in range(n)) + ']>')
    array = '!array.type<2 x !P>'
    print('!W = !pod.type<[@pod: !P]>')
    name = 'RangedCopyScale'
    arguments = f'%array: {array}, %index: index'
print(f'''module attributes {{llzk.lang}} {{
  struct.def @{name} {{
    struct.member @first : !felt.type
    struct.member @last : !felt.type
    function.def @compute({arguments}) -> !struct.type<@{name}> {{
      %self = struct.new : !struct.type<@{name}>
      %seven = felt.const 7''')
if args.kind == 'concrete':
    for i in range(n):
        print(f'      %c{i} = arith.constant {i} : index')
        print(f'      array.write %array[%c{i}] = %seven : {array}, !felt.type')
    print(f'''      %copy = pod.new {{ @array = %array }} : !W
      %copied = pod.read %copy[@array] : !W, {array}
      %first = array.read %copied[%c0] : {array}, !felt.type
      %last = array.read %copied[%c{n-1}] : {array}, !felt.type''')
else:
    print(f'''      %nine = felt.const 9
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %left = array.read %array[%zero] : {array}, !P
      %right = array.read %array[%one] : {array}, !P''')
    for i in range(n):
        print(f'      pod.write %left[@field{i}] = %seven : !P, !felt.type')
        print(f'      pod.write %right[@field{i}] = %nine : !P, !felt.type')
    print(f'''      %selected = array.read %array[%index] : {array}, !P
      %copy = pod.new {{ @pod = %selected }} : !W
      %copied = pod.read %copy[@pod] : !W, !P
      %first = pod.read %copied[@field0] : !P, !felt.type
      %last = pod.read %copied[@field{n-1}] : !P, !felt.type''')
print(f'''      struct.writem %self[@first] = %first : !struct.type<@{name}>, !felt.type
      struct.writem %self[@last] = %last : !struct.type<@{name}>, !felt.type
      function.return %self : !struct.type<@{name}>
    }}
    function.def @constrain(%self: !struct.type<@{name}>, {arguments}) {{
      function.return
    }}
  }}
}}''')
