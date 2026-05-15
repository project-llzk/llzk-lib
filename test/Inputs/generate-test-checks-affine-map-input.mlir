#map = affine_map<(d0) -> (d0)>
module {
  func.func @f(%arg0: index) -> index {
    %0 = affine.apply #map(%arg0)
    return %0 : index
  }
}
