import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def FEA(nelx, nely, x, penal):
    # dofs:
    ndof = 2*(nelx+1)*(nely+1)
    xPhys = x.copy()
    KE = lk()
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely
            n1 = (nely+1)*elx+ely
            n2 = (nely+1)*(elx+1)+ely
            edofMat[el, :] = np.array(
                #[2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
                [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3 ])
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    # BC's and support
    dofs = np.arange(2*(nelx+1)*(nely+1))
    #fixed = np.union1d(dofs[0:2*(nely+1):2], np.array([2*(nelx+1)*(nely+1)-1]))
    fixed = dofs[0:2*(nely+1)]
    free = np.setdiff1d(dofs, fixed)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # Set load
    #f[1, 0] = -1
    
    #f[ndof-1,0] = -1
    
    # CASE 1
    #loaddof=np.array(range(2*(nely+1)*nelx,2*(nely+1)*(nelx+1),2))
    #f[loaddof,0]=1
    
    # CASE 2
    loaddof=np.array(range(2*(nely+1)*nelx+2,2*(nely+1)*(nelx+1)-2,2))
    f[loaddof,0]=1
    f[2*(nely+1)*nelx,0]=0.5
    f[2*(nely+1)*(nelx+1)-2,0]=0.5
    
    
    #sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)
    #                                    ** penal*(Emax-Emin))).flatten(order='F')
    sK = ((KE.flatten()[np.newaxis]).T*((xPhys)** penal)).flatten(order='F')
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    # Remove constrained dofs from matrix
    K = K[free, :][:, free]
    # Solve system
    u[free, 0] = spsolve(K, f[free, 0])
    return u
def CompuStress(nelx, nely, x, penal, u):
    D=ElasticTensor()
    B=StrainDispMat()
    
    SMat = np.zeros([(nelx+1)*(nely+1),3])
    NumCount = np.zeros([(nelx+1)*(nely+1),1])
    strs = np.zeros([(nelx+1)*(nely+1)*3,1])
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely
            n1 = (nely+1)*elx+ely
            n2 = (nely+1)*(elx+1)+ely
            edof = np.array(
                [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3 ])
            ue=u[edof,0]
            n3 = n2 + 1
            n4 = n1 + 1
            SMat[n1,:]=SMat[n1,:]+x[el]**penal*D@B[0,:,:]@ue
            SMat[n2,:]=SMat[n2,:]+x[el]**penal*D@B[1,:,:]@ue
            SMat[n3,:]=SMat[n3,:]+x[el]**penal*D@B[2,:,:]@ue
            SMat[n4,:]=SMat[n4,:]+x[el]**penal*D@B[3,:,:]@ue
            NumCount[[n1,n2,n3,n4],0]=NumCount[[n1,n2,n3,n4],0]+[1,1,1,1]
    for i in range((nelx+1)*(nely+1)):
        strs[[i*3,i*3+1,i*3+2],0]=SMat[i,:]/NumCount[i]
    return strs
#element stiffness matrix
def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
	return (KE)
def ElasticTensor():
    E=1
    nu=0.3
    D = E/(1-nu**2)*np.array([[1,nu,0],
                  [nu,1,0],
                  [0,0,(1-nu)/2] ])
    return D
def StrainDispMat():
    B=np.zeros([4,3,8],dtype=float)
    B[0,:,:]=np.array([ [-1,  0, 1, 0, 0, 0, 0, 0],
                        [0, -1, 0, 0, 0, 0, 0, 1],
                        [-1, -1, 0, 1, 0, 0, 1, 0] ])
    B[1,:,:]=np.array([ [-1,  0,  1,  0, 0, 0, 0, 0],
                        [0,  0,  0, -1, 0, 1, 0, 0],
                        [0, -1, -1,  1, 1, 0, 0, 0] ])
    B[2,:,:]=np.array([ [0, 0,  0,  0, 1, 0, -1,  0],
                        [0, 0,  0, -1, 0, 1,  0,  0],
                        [0, 0, -1,  0, 1, 1,  0, -1] ])
    B[3,:,:]=np.array([ [0,  0, 0, 0, 1, 0, -1,  0],
                        [0, -1, 0, 0, 0, 0,  0,  1],
                        [-1,  0, 0, 0, 0, 1,  1, -1] ])
    return B

def ComputeTarget(input):
    # x np.array [bs, 1, 20, 40]
    nelx = 40
    nely = 20
    penal = 3.0
    bs = input.shape[0]
    result = np.zeros([bs, 5, 21, 41], dtype=float)
    data = input.transpose([0,1,3,2]).reshape(-1,800)
    # data = input.permute([0,1,3,2]).contiguous().view(-1,800)
    for i in range(bs):
        xx = data[i]
        u = FEA(nelx, nely, xx, penal)
        strs=CompuStress(nelx, nely, xx, penal, u)
        ux = u[range(0,1722,2)]
        uy = u[range(1,1722,2)]
        sx = strs[range(0,2583,3)]
        sy = strs[range(1,2583,3)]
        sxy = strs[range(2,2583,3)]
        result[i] = np.concatenate([ux,uy,sx,sy,sxy],1).T.reshape(5,41,21).transpose([0,2,1])
    return result


if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = 20
    penal = 3.0
    volfrac = 0.4
    x = volfrac * np.ones(nely*nelx, dtype=float)
    # 0.001 <= x <= 1
    u = FEA(nelx, nely, x, penal)
    print(u[42:84:1])
    #print(u[43:84:2])
    strs=CompuStress(nelx, nely, x, penal, u)
    print(strs[0:14:1])