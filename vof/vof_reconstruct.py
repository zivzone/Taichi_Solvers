from vof_data import *

@ti.kernel
def reconstruct_plic():
	for i,j,k in C:
		if Flags[i,j,k]&cellFlags.CELL_INTERFACE==cellFlags.CELL_INTERFACE:
			if (i>0 and j>0 and k>0 and i<n_x-1 and j<n_y-1 and k<n_z-1):
				mx,my,mz,alpha = recon(i,j,k)
				M[i,j,k][0] = mx
				M[i,j,k][1] = my
				M[i,j,k][2] = mz
				Alpha[i,j,k] = alpha

@ti.func
def calc_C(alpha, m):
  # should store the following for speedup
  # convert normal vector into Zaleski's m vector
	c = 0.0
	if (alpha < 0.0):
		c = 0.0
	elif (alpha > ti.abs(m[0])+ti.abs(m[1])+ti.abs(m[2])):
		c = 1.0

	a = ti.min(alpha, ti.abs(m[0]) + ti.abs(m[1])+ ti.abs(m[2]) - alpha)
	mx = ti.abs(m[0])
	my = ti.abs(m[1])
	mz = ti.abs(m[2])

	# the coefficients of the normal must be ordered as: m1 < m2 < m3
	m1 = ti.min(mx,my)
	m3 = ti.max(ti.max(mx,my),1.0e-15)
	m2 = mz
	if (m2 < m1):
		tmp = m1
		m1 = m2
		m2 = tmp
	elif (m2 > m3):
		tmp = m3
		m3 = m2
		m2 = tmp

	m12 = m1 + m2
	mm  = ti.min(m12,m3)
	pr  = ti.max(6.0*m1*m2*m3,1.0e-50)
	V1  = m1*m1*m1/pr

	if (a <  m1):
		c = a*a*a/pr
	elif (a < m2):
		c = 0.5*a*(a-m1)/(m2*m3)+V1
	elif (a < mm):
		c = (a*a*(3.0*m12-a)+m1*m1*(m1-3.0*a)+m2*m2*(m2-3.0*a))/pr
	elif (m12 <= m3):
		c = (a-0.5*m12)/m3
	else:
		c = (a*a*(3.0*(m1+m2+m3)-2.0*a) + m1*m1*(m1-3.0*a) + \
		m2*m2*(m2-3.0*a) + m3*m3*(m3-3.0*a))/pr

	if (alpha > 0.5*ti.abs(m[0])+ti.abs(m[1])+ti.abs(m[2])):
		c = 1.0-c

	return c

@ti.func
def calc_lsq_vof_error(alpha, m, i, j, k):
	error = 0.0
	for dk in range(-1,2):
		for dj in range(-1,2):
			for di in range(-1,2):
				# skip central cell, since the error there should be zero
				#if (dk==0 and dj==0 and di==0):
				#	continue
				a = alpha - (m[0]*di + m[1]*dj + m[2]*dk)
				error = error + (C[i+di,j+dj,k+dk] - calc_C(a,m))
	return error


@ti.func
def my_cbrt(n):
	iter = 0
	root = 1.0
	if n>1:
		a = 0.0
		b = n
		root = (a+b)/2.0
		while (root*root*root-n >1e-8 or iter < 100):
			root = (a+b)/2.0
			if root*root*root<n:
				a = root
			else:
				b = root
			iter = iter + 1
	elif n<1:
		a = 1.0
		b = n
		root = (a+b)/2.0
		while (root*root*root-n >1e-8 or iter < 100):
			root = (a+b)/2.0
			if root*root*root>n:
				a = root
			else:
				b = root
			iter = iter + 1

	return root


@ti.func
def calc_alpha(c, m):
  # reconstruct interface as line/plane
  # for 3D: use S. Zaleski's Surfer code routine al3d:
  #         find alpha IN: m1 x1 + m2 x2 + m3 x3 = alpha, given m1+m2+m3=1 (all > 0) and VoF value
  #         assumes that cell is unit size, i.e. alpha is relative to dx=dy=dz=1
  #         Note: alpha is not with respect to the lower,front,left corner of the cell. To get it for
  #         this "standard" coordinate system, coordinate mirroring (corrections to alpha) would have to
  #---------------------------------------
	alpha = 0.0

	r13 = 1.0/3.0
	if (c <= Czero or c >= Cone):
		alpha = -1.0e10
	else:
		# convert normal vector into Zaleski's m vector
		mx = ti.abs(m[0])
		my = ti.abs(m[1])
		mz = ti.abs(m[2])

		# the coefficients of the normal must be ordered as: m1 < m2 < m3
		m1 = ti.min(mx,my)
		m3 = ti.max(ti.max(mx,my),1.0e-15)
		m2 = mz
		if (m2 < m1):
			tmp = m1
			m1 = m2
			m2 = tmp
		elif (m2 > m3):
			tmp = m3
			m3 = m2
			m2 = tmp

		# get ranges: V1<V2<v3;
		m12 = m1 + m2
		pr  = ti.max(6.0*m1*m2*m3,1.0e-20)
		V1  = m1*m1*m1/pr
		V2  = V1 + 0.5*(m2-m1)/m3
		V3  = 0.0
		mm = 0.0
		if (m3 < m12):
			mm = m3
			V3 = ( m3*m3*(3.0*m12-m3) + m1*m1*(m1-3.0*m3) + m2*m2*(m2-3.0*m3) )/pr
		else:
			mm = m12
			V3 = 0.5*mm/m3

		# limit ch (0.d0 < ch < 0.5d0);
		ch = ti.min(c,1.0-c);

		# calculate d
		if (ch < V1):
			alpha = my_cbrt(pr*ch) # my own cube root function since taichi doesnt have one yet
		elif (ch < V2):
			alpha = 0.5*(m1 + ti.sqrt(m1*m1 + 8.0*m2*m3*(ch-V1)))
		elif (ch < V3):
			p = 2.0*m1*m2
			q = 1.5*m1*m2*(m12 - 2.0*m3*ch)
			p12 = ti.sqrt(p)
			teta = ti.acos(q/(p*p12))*r13
			cs = ti.cos(teta)
			alpha = p12*(ti.sqrt(3.0*(1.0-cs*cs)) - cs) + m12
		elif (m12 <= m3):
			alpha = m3*ch + 0.5*mm
		else:
			p = m1*(m2+m3) + m2*m3 - 0.25*(m1+m2+m3)*(m1+m2+m3)
			q = 1.5*m1*m2*m3*(0.5-ch)
			p12 = ti.sqrt(p)
			teta = ti.acos(q/(p*p12))*r13
			cs = ti.cos(teta)
			alpha = p12*(ti.sqrt(3.0*(1.0-cs*cs)) - cs) + 0.5*(m1+m2+m3)

		if (c > 0.5):
			alpha = (m1+m2+m3)-alpha

	return alpha


@ti.func
def ELVIRA(i, j, k):
	# reconstruct planar interface using ELVIRA
	# check all possible normal vectors using forward and backward differrences
	h = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
	hx = [0.0, 0.0, 0.0]
	hy = [0.0, 0.0, 0.0]
	hz = [0.0, 0.0, 0.0]
	n = [0.0,0.0,0.0]
	m = [0.0,0.0,0.0]
	alpha = 0.0

	errorMin = 1.0e8

	# x-heights
	for dk in ti.static(range(-1,2)):
		for dj in ti.static(range(-1,2)):
			h[dj+1][dk+1] = C[i-1,j+dj,k+dk] + C[i,j+dj,k+dk] + C[i+1,j+dj,k+dk]

	# forward, central, backward difference
	hy[0] = (h[1][1] - h[0][1])
	hy[1] = (h[2][1] - h[0][1])*.5
	hy[2] = (h[2][1] - h[1][1])

	hz[0] = (h[1][1] - h[1][0])
	hz[1] = (h[1][2] - h[1][0])*.5
	hz[2] = (h[1][2] - h[1][1])

	# loop over all possible difference
	for kk in ti.static(range(3)):
		for jj in ti.static(range(3)):
			n[1] = hy[jj]
			n[2] = hz[kk]
			n[0] = 1.0
			if (C[i+1,j,k] - C[i-1,j,k] < 0.0):
				n[0] = -1.0

			# make sum of components = 1 for PLIC reconstruction and reconstruct
			rdenom = 1.0/(ti.max(ti.abs(n[0]) + ti.abs(n[1]) + ti.abs(n[2]) , 1.0e-8))
			n[0] = -n[0]*rdenom
			n[1] = -n[1]*rdenom
			n[2] = -n[2]*rdenom
			alp =  calc_alpha(C[i,j,k], n)
			error = 1.0#calc_lsq_vof_error(alp,n,i,j,k)

			if (error < errorMin):
				errorMin = error
				alpha = alp
				m[0] = n[0]
				m[1] = n[1]
				m[2] = n[2]

	# y-heights
	for dk in ti.static(range(-1,2)):
		for di in ti.static(range(-1,2)):
			h[di+1][dk+1] = C[i+di,j-1,k+dk] + C[i+di,j,k+dk] + C[i+di,j+1,k+dk]

	# forward, central, backward difference
	hx[0] = (h[1][1] - h[0][1])
	hx[1] = (h[2][1] - h[0][1])*.5
	hx[2] = (h[2][1] - h[1][1])

	hz[0] = (h[1][1] - h[1][0])
	hz[1] = (h[1][2] - h[1][0])*.5
	hz[2] = (h[1][2] - h[1][1])

	# loop over all possible difference
	for kk in ti.static(range(3)):
		for ii in ti.static(range(3)):
			n[0] = hx[ii]
			n[2] = hz[kk]
			n[1] = 1.0
			if (C[i+1,j,k] - C[i-1,j,k] < 0.0):
				n[1] = -1.0

			# make sum of components = 1 for PLIC reconstruction and reconstruct
			rdenom = 1.0/(ti.max(ti.abs(n[0]) + ti.abs(n[1]) + ti.abs(n[2]) , 1.0e-8))
			n[0] = -n[0]*rdenom
			n[1] = -n[1]*rdenom
			n[2] = -n[2]*rdenom
			alp = calc_alpha(C[i,j,k], n)
			error = calc_lsq_vof_error(alp,n,i,j,k)

			if (error < errorMin):
				errorMin = error
				alpha = alp
				m[0] = n[0]
				m[1] = n[1]
				m[2] = n[2]

	# z-heights
	for dj in ti.static(range(-1,2)):
		for di in ti.static(range(-1,2)):
			h[di+1][dj+1] = C[i+di,j+dj,k-1] + C[i+di,j+dj,k] + C[i+di,j+dj,k+1]

	# forward, central, backward difference
	hx[0] = (h[1][1] - h[0][1])
	hx[1] = (h[2][1] - h[0][1])*.5
	hx[2] = (h[2][1] - h[1][1])

	hy[0] = (h[1][1] - h[1][0])
	hy[1] = (h[1][2] - h[1][0])*.5
	hy[2] = (h[1][2] - h[1][1])

	# loop over all possible difference
	for jj in ti.static(range(3)):
		for ii in ti.static(range(3)):
			n[0] = hx[ii]
			n[1] = hy[jj]
			n[2] = 1.0
			if (C[i+1,j,k] - C[i-1,j,k] < 0.0):
				n[2] = -1.0

			# make sum of components = 1 for PLIC reconstruction and reconstruct
			rdenom = 1.0/(ti.max(ti.abs(n[0]) + ti.abs(n[1]) + ti.abs(n[2]) , 1.0e-8));
			n[0] = -n[0]*rdenom
			n[1] = -n[1]*rdenom
			n[2] = -n[2]*rdenom
			alp = calc_alpha(C[i,j,k], n)
			error = calc_lsq_vof_error(alp,n,i,j,k)

			if (error < errorMin):
				errorMin = error
				alpha = alp
				m[0] = n[0]
				m[1] = n[1]
				m[2] = n[2]

	return m[0], m[1], m[2], alpha


# set the reconstrunction function
recon = ELVIRA
