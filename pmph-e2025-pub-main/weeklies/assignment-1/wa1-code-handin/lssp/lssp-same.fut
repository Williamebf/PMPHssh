-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input { [1, -2, -2, 0, 0, 0, 1, 2, 3, 4, -6,] }   output {3}
-- compiled input { [1,2,3,4] }                                   output {1}
-- compiled input { [5,4,3,2,1,0] }                                   output {1}
-- compiled input { [2,1,3] }                                   output {1}
-- compiled input { [-5,-1, 0,0, 1, 1, 2] }                          output {2}
-- compiled input { [-1, 0, 0, 1, 1, 2] }                           output {2}
-- compiled input { [1, 2, 3, -1, -1, 3, 1, 2] }                  output {2}
-- compiled input { [1, 3, 7, 3, 4, 1, 1, 3] }                  output {2}
-- compiled input { [-1, 3, 1, 2, 3, 1, 4, 0] }                  output {1}
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }
-- compiled input @ random_i32_array.txt auto outout
-- compiled input @ random2_i32_array.txt auto outout
-- compiled input @ random3_i32_array.txt output {16777216}

import "lssp"
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
  --in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
