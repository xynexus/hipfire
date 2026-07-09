---
title: "Area Group Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Area-Group-Constraint"
toc_id: Dbf_OdaGze9qBWt7sraNXg
content_id: ZRsRTryawUieX5tda8YvKg
---

##### Area Group Constraint

The area group constraint specifies a range of tile and/or shim locations to which a group of one or more nodes can be mapped. You can specify the area group constraint with the following properties.

- **groups:** Specify the collection of groups. The groups can be:
- **groups:** tile-type Specify the tile-type for the group. Supported tile-types are aie_tile, shim_tile, or memory_tile. column_min Column index for lower left corner of the group. row_min Row index for lower left corner of the group. column_max Column index for upper right corner of the group. row_max Row index for upper right corner of the group.
- **contain_routing:** A boolean value that when set to true, ensures all routing, including nets between nodes contained in the nodeGroup, is contained within the area group.
- **exclude_routing:** A boolean value that when set to true, ensures all routing, excluding nets between nodes from the nodeGroup, is excluded from the area Group.
- **exclude_placement:** A boolean value that when specified true prevents all nodes not included in nodeGroup from being placed within the area group bounding box.

An AI Engine tile or memory tile range is in the form (column, row):(column, row). The first tile is the bottom left corner and the second tile the upper right corner. The column and row values are zero based, where the zeroth row is counted from the bottom-most compute processor and the zeroth column is counted from the left-most column.

A shim range is in the form of (column):(column). The first value is the left-most column and the second value the right-most column. The column is zero based, where the zeroth row is counted from the bottom-most compute processor and the zeroth column is counted from the left-most column. You can also specify an optional channel for the shim range, for example, (column, channel):(column, channel).

You can use the area group to exclude a range on the device from being used by the compiler for mapping and optionally routing as follows:

- To exclude a range from router, set nodeGroup and set exclude_placement to true.
- To exclude a range from mapper, set nodeGroup and set exclude_routing to true.
- To include all nets within a nodeGroup for router and mapper, set nodeGroup and set contain_routing to true.

**Note:** There can be any number of area group constraints in the Global Constraints section, as long as each constraint has a unique name.

###### Syntax

```
"areaGroup": {
  "name": string,

  "nodeGroup": [<node list>], (*optional)
  "tileGroup": [<tile list>], (*optional)
  "shimGroup": [<shim list>], (*optional)
  "contain_routing": bool, (*optional)
  "exclude_routing": bool, (*optional)
  "exclude_placement"" bool, (*optional)
}

<node list> ::= <node name>[,<node name>...]

<tile array> ::= <tile value>[,<tile value>...]
<tile value> ::= <tile range> | <tile address>
<tile range> ::= "<tile address>[:<tile address>]"
<tile address> ::= (<column>, <row>)

<shim array> ::= <shim value>[,<shim value>...]
<shim value> ::= <shim range> | <shim address>
<shim range> ::= "<shim address>[:<shim address>]"
<shim address> ::= (<column>[,<channel>])

<node name> ::= string
<column> ::= integer
<row> ::= integer
<channel> ::= integer
```

###### Example

```
{
  "GlobalConstraints": {
    "areaGroup": {
      "name": "mygraph_area_group",
      "nodeGroup": ["mygraph.k1", "mygraph.k2"],
      "tileGroup": ["(2,0):(2,3)"],
      "shimGroup": ["0:3"]
    }
  }
}
```

###### Example of Exclude

```
{
  "GlobalConstraints": {
    "areaGroup": {
      "name": "mygraph_excluded_area_group",

      "tileGroup": ["(3,0):(4,3)"],
      "shimGroup": ["3:4"],
      "exclude_routing": true
    }
  }
}
```
