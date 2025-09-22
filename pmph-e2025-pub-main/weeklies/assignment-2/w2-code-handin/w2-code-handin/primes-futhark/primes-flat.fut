-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }
-- output @ ref10000000.out
let sgmScan [n] 't
            (op: t -> t -> t)
            (ne: t)
            (flags: [n]bool)
            (vals: [n]t)
            : [n]t =
  scan (\(f1, v1) (f2, v2) -> (f1 || f2, if f2 then v2 else op v1 v2))
       (false, ne)
       (zip flags vals)
  |> unzip
  |> (.1)
let mkFlagArray 't [m] 
            (aoa_shp: [m]i64) (zero: t)   --aoa_shp=[0,3,1,0,4,2,0]
            (aoa_val: [m]t  ) : []t   =   --aoa_val=[1,1,1,1,1,1,1]
  let shp_rot = map (\i->if i==0 then 0   --shp_rot=[0,0,3,1,0,4,2]
                         else aoa_shp[i-1]
                    ) (iota m)
  let shp_scn = scan (+) 0 shp_rot       --shp_scn=[0,0,3,4,4,8,10]
  let aoa_len = if m == 0 then 0         --aoa_len= 10
                else shp_scn[m-1]+aoa_shp[m-1]
  let shp_ind = map2 (\shp ind ->        --shp_ind= 
                       if shp==0 then -1 --  [-1,0,3,-1,4,8,-1]
                       else ind          --scatter
                     ) aoa_shp shp_scn   --   [0,0,0,0,0,0,0,0,0,0]
  in scatter (replicate aoa_len zero)    --   [-1,0,3,-1,4,8,-1]
             shp_ind aoa_val             --   [1,1,1,1,1,1,1]
                                     -- res = [1,0,0,1,1,0,0,0,1,0] 

let sgmSumF32 [n] (flags: [n]bool) (vals: [n]f32) : [n]f32 =
  sgmScan (+) 0f32 flags vals
  
let primesFlat (n: i64) = --: []i64 =
  let sq_primes = [2i64, 3i64, 5i64, 7i64]
  let len = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      -- if n/len < len then n = 8, else 64
      
      let len = if n / len < len then n else len*len
      
      -- map prime 8/prime-1 = [8/2-1 = 3, 8/3-1 = 1, 8/5-1 = 0, 8/7 -1= 0]
      let mult_lens = map (\p -> (len / p) - 1) sq_primes
      let flen = reduce (+) 0 mult_lens
      -- Sum 3+1 = 4
      -- mn = multlen
      -- p = sqrt_primes
      -- Scan inclusive
      -- [3, 4, 4, 4]
      -- create flag
      let flagtemp = mkFlagArray mult_lens 0 mult_lens 

      let flagtemp = flagtemp :> [flen]i64
      let vals = map (\f -> if f!= 0 then 0 else 1) flagtemp
      let iotsp1 = sgmScan (+) 0 (map bool.i64 flagtemp) vals
      let twom = map (+2) iotsp1
      let (flag_n, flag_v) = zip mult_lens sq_primes |> mkFlagArray mult_lens (0,0) |> unzip 
      let rp = sgmScan (+) 0 (map bool.i64 flag_n) flag_v
      let not_primes = map2(\j p -> j*p) twom (rp :> [flen]i64)
      --------------------------------------------------------------
      -- The current iteration knows the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the correct `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`. 
      
      --let not_primes = replicate flat_size 0
      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------
      -- [0,0,0,0]
       let zero_array  = replicate flen 0i8
      -- [0,0,1,1,1,1,1,1]
       let mostly_ones = map (i8.bool <-< (> 1)) (iota (len + 1))
      -- scatter mostly primes i [0,0,primes]
       let prime_flags = scatter mostly_ones not_primes zero_array
       -- unsafe flag disables bounds checking for prime flag array.
       -- Rungs over primeflags, puts i into index, if prime flag != 0
       let sq_primes   = filter (\i -> #[unsafe] prime_flags[i] != 0) (iota (len + 1))
       in (sq_primes, len)
  in sq_primes

-- RUN a big test with:
--   $ futhark cuda primes-flat.fut
--   $ echo "10000000i64" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
-- or simply use futhark bench, i.e.,
--   $ futhark bench --backend=cuda primes-flat.fut
let main (n: i64) = primesFlat n
-- : []i64