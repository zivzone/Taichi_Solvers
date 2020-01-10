from vof_data import *
from vof_util import *

@ti.kernel
def copy_to_temp():
  for i,j,k in Flags:
    C_temp[i,j,k] = C[i,j,k]
    Flags_temp[i,j,k] = Flags[i,j,k]

@ti.kernel
def copy_from_temp():
  for i,j,k in Flags_temp:
    C[i,j,k] = C_temp[i,j,k]
    Flags[i,j,k] = Flags_temp[i,j,k]

@ti.kernel
def grow_band():
  for i,j,k in Flags_temp:
    if is_internal_cell(i,j,k) and (Flags_temp[i,j,k]&flag_enum.CELL_ACTIVE)==flag_enum.CELL_ACTIVE: #use Flags_temp
      # check if this is and interface cell
      if (C_temp[i,j,k] >= Czero and C_temp[i,j,k] <= Cone):
        Flags[i,j,k] = flag_enum.CELL_INTERFACE
        C[i,j,k] = C_temp[i,j,k]
      else:
        # treat cases where interface sits on cell face
        if (C_temp[i,j,k] > Cone):
          if (C_temp[i-1,j,k] < Czero or C_temp[i,j-1,k] < Czero or C_temp[i,j,k-1] < Czero):
            Flags[i,j,k] = flag_enum.CELL_INTERFACE
            C[i,j,k] = C_temp[i,j,k]
        elif (C_temp[i,j,k] < Czero):
          if (C_temp[i-1,j,k] > Cone or C_temp[i,j-1,k] > Cone or C_temp[i,j,k-1] > Cone):
            Flags[i,j,k] = flag_enum.CELL_INTERFACE
            C[i,j,k] = C_temp[i,j,k]

  for i,j,k in Flags:
    if is_internal_cell(i,j,k) and is_interface_cell(i,j,k):
      # flag this cell and neighbors as active
      for dk in ti.static(range(-1,2)):
        for dj in ti.static(range(-1,2)):
          for di in ti.static(range(-1,2)):
            Flags[i+di,j+dj,k+dk] = Flags[i+di,j+dj,k+dk]|flag_enum.CELL_ACTIVE
            C[i+di,j+dj,k+dk] = C_temp[i+di,j+dj,k+dk]

  for i,j,k in Flags:
    if is_internal_cell(i,j,k) and is_active_cell(i,j,k):
      # flag all faces of this cell as active
      Flags[i,j,k] = Flags[i,j,k]|flag_enum.X_FACE_ACTIVE
      Flags[i+1,j,k] = Flags[i+1,j,k]|flag_enum.X_FACE_ACTIVE

  for i,j,k in Flags:
    if is_internal_cell(i,j,k) and is_active_cell(i,j,k):
      # flag all faces of this cell as active
      Flags[i,j,k] = Flags[i,j,k]|flag_enum.Y_FACE_ACTIVE
      Flags[i,j+1,k] = Flags[i,j+1,k]|flag_enum.Y_FACE_ACTIVE

  for i,j,k in Flags:
    if is_internal_cell(i,j,k) and is_active_cell(i,j,k):
      # flag all faces of this cell as active
      Flags[i,j,k] = Flags[i,j,k]|flag_enum.Z_FACE_ACTIVE
      Flags[i,j,k+1] = Flags[i,j,k+1]|flag_enum.Z_FACE_ACTIVE

  for i,j,k in Flags:
    if is_internal_cell(i,j,k) and is_active_cell(i,j,k):
      # flag neighbors of active cells as buffer cells
      for dk in ti.static(range(-1,2)):
        for dj in ti.static(range(-1,2)):
          for di in ti.static(range(-1,2)):
            if is_active_cell(i+di,j+dj,k+dk)==False:
              Flags[i+di,j+dj,k+dk] = Flags[i+di,j+dj,k+dk]|flag_enum.CELL_BUFFER
              C[i+di,j+dj,k+dk] = C[i,j,k]
