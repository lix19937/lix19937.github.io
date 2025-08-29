#!/bin/bash

tegrastats --interval 500|  awk '{print $1" "$2 " " $9"@"$10 " " $19}' 
