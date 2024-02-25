#!/bin/sh

rm -f $2
while IFS=, read a b; do
    echo "$((a + b))" >> $2
done < $1

echo "end of my work"