-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input { [1, -2, -2, 0, 0, 0, 1, 2, 3, 4, -6,] }   output {3}
-- compiled input { [1,2,3,4] }                                   output {0}
-- compiled input { [5,4,3,2,1,0] }                                   output {1}
-- compiled input { [2,1,3] }                                   output {0}
-- compiled input { [-5,-1, 0,0, 1, 1, 2] }                          output {2}
-- compiled input { [-1, 0, 0, 1, 1, 2] }                           output {2}
-- compiled input { [1, 2, 3, -1, -1, 3, 1, 2] }                  output {0}
-- compiled input { [1, 3, 7, 3, 4, 1, 1, 3] }                  output {0}
-- compiled input { [-1, 3, 1, 2, 3, 1, 4, 0] }                  output {1}
-- compiled input {
--    [1i32, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }
-- compiled input @ random_i32_array.txt auto output
-- compiled input @ random2_i32_array.txt auto output
-- compiled input @ random3_i32_array.txt output {0}

import "lssp-seq"
import "lssp"

type int = i32

let main (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
  --in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
