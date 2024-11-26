import numpy as np

JAP_MEL_VEC = np.array([0.4143, 0.3570, 0.8372])
JAP_HEM_VEC = np.array([0.2988, 0.6838, 0.6657])

jap_pigments = {"mel": JAP_MEL_VEC, "hem": JAP_HEM_VEC}


def pigment_separation(img, melanin=JAP_MEL_VEC, hemoglobin=JAP_HEM_VEC):
    # {{{
    """[1] Set dye vector"""
    # melanin vector and hemoglobin vector
    # melanin = np.array([0.4143, 0.3570, 0.8372])
    # hemoglobin = np.array([0.2988, 0.6838, 0.6657])

    """ [2] Get/set image information """
    height, width, channels = img.shape[:3]
    Img_size = height * width

    """ [3] Acquisition of normal of skin color distribution plane """
    # normal of skin color distribution plane = outer product of two color
    # vectors
    # Formula to find the normal from the plane (vec(1,:) = [1 1 1] so not
    # considered
    # in the formula)
    housen = [
        melanin[1] * hemoglobin[2] - melanin[2] * hemoglobin[1],
        melanin[2] * hemoglobin[0] - melanin[0] * hemoglobin[2],
        melanin[0] * hemoglobin[1] - melanin[1] * hemoglobin[0],
    ]

    """ [4] Array initialization, data shaping """
    DC = 1 / 255.0
    L = np.zeros((height * width * 3, 1))
    linearSkin = np.zeros((3, Img_size))
    S = np.zeros((3, Img_size))

    img = img.copy()[:, :, ::-1]

    img_r = np.reshape(img[:, :, 0].T, Img_size)
    img_g = np.reshape(img[:, :, 1].T, Img_size)
    img_b = np.reshape(img[:, :, 2].T, Img_size)

    skin = np.array([img_r[:], img_g[:], img_b[:]])

    """ [5] Image ~gamma correction~ (normalize image maximum to 1) """
    for j in range(3):
        linearSkin[j] = skin[j, :].astype(np.float32) / 255

    # create mask image
    img_mask = np.where(linearSkin == 0, 0, 1)
    img_mask2 = np.where(linearSkin == 0, DC, 0)

    """ [6] To density space (log space) """
    linearSkin = linearSkin + img_mask2
    S = -np.log(linearSkin)
    S = S * img_mask.astype(np.float64)

    """ [7] Start of skin color space to 0 """
    S = S - np.array([[0, 0, 0]]).T

    """ [8] Find the intersection of a straight line passing through a
    component parallel to the direction of illumination unevenness and the skin
    color distribution plane."""
    # housen: Normal of skin color distribution plane
    # S: RGB in density space
    # vec: independent component vector
    t = -(
        np.dot(housen[0], S[0])
        + np.dot(housen[1], S[1])
        + np.dot(housen[2], S[2])
    ) / (housen[0] + housen[1] + housen[2])

    # shadow removal
    # skin_flat: the plane of the unshaded melahemo vector
    # rest: shadow component
    skin_flat = t[:, np.newaxis].T + S
    rest = S - skin_flat

    """ [9] Dye Concentration Calculation """
    # ------------------------------------------------- ------------
    # mixing and separating matrices
    CompSynM = np.array([melanin, hemoglobin]).T
    CompExtM = np.linalg.pinv(CompSynM)
    # get dye density for each pixel
    # Density distribution = [melanin pigment; hemoglobin pigment]
    # = Skin color vector (after shading removal) x Separation matrix
    Component = np.dot(CompExtM, skin_flat)

    # Correction of hemoglobin component (because it becomes a negative number)
    # Component(2,:) = Component(2,:) + 0;
    # ------------------------------------------------- ------------
    Comp = np.vstack((Component, rest[0, :][np.newaxis, :]))

    L = np.hstack([Comp[0, :], Comp[1, :], Comp[2, :]])

    L_Mel = L[0:Img_size]
    L_Hem = L[Img_size : Img_size * 2]
    L_Shd = L[Img_size * 2 :]

    return (L_Mel, L_Hem, L_Shd)


# }}}


def hem_separation(img, melanin=JAP_MEL_VEC, hemoglobin=JAP_HEM_VEC):
    # {{{
    """
    [10] Transformation of color components into image space
    Returns just the Hem map
    """
    _, L_Hem, _ = pigment_separation(img, melanin, hemoglobin)

    height, width, channels = img.shape[:3]

    # Mask3 = (np.sum(img, axis=2) > 0).astype(np.uint8)

    # get dye image
    img2 = L_Hem.reshape(width, height).T
    img_e = np.exp(-img2)
    img_exp = np.clip(img_e, 0, 1)
    ef_img = img_exp.astype(np.float32)

    return ef_img


# }}}


def hem_separation_img(img, melanin=JAP_MEL_VEC, hemoglobin=JAP_HEM_VEC):
    # {{{
    """
    [10] Transformation of color components into image space
    Returns a BGR image of the Hem channel
    """

    L_Mel, L_Hem, L_Shd = pigment_separation(img, melanin, hemoglobin)

    height, width, channels = img.shape[:3]
    Img_size = height * width

    # This is different from the original implementation, but I believe this
    # was the actual intent of the implementation
    # Mask3 = np.empty((height, width, 3), dtype=np.bool)
    # Mask3[:, :, 0] = np.sum(img, axis=2) > 0
    # Mask3[:, :, 1] = np.sum(img, axis=2) > 0  # original: img[..., 0] > 0
    # Mask3[:, :, 2] = np.sum(img, axis=2) > 0  # original: img[..., 0] > 0

    # get dye image
    L_Obj = np.zeros((3, Img_size))

    for j in range(3):
        L_Obj[j, :] = hemoglobin[j] * L_Hem

    L_Vec = np.hstack([L_Obj[0, :], L_Obj[1, :], L_Obj[2, :]])

    img2 = np.reshape(L_Vec, (3, width, height)).T
    img_e = np.exp(-img2)
    img_exp = np.clip(img_e, 0, 1)[:, :, ::-1]
    # ef_img = img_exp * Mask3.astype(np.float32)
    ef_img = img_exp.astype(np.float32)

    return ef_img


# }}}


def pigment_channels(img, melanin=JAP_MEL_VEC, hemoglobin=JAP_HEM_VEC):
    # {{{
    """
    [10] Transformation of color components into image space
    Returns a 3 channel image where ch0 is Melanin, ch1 is Hemoglobin and ch2
    is shade
    """

    L_Mel, L_Hem, L_Shd = pigment_separation(img, melanin, hemoglobin)

    height, width, channels = img.shape[:3]
    Img_size = height * width

    # Mask3 = np.empty((height, width, 3), dtype=np.bool)
    # Mask3[:, :, 0] = np.sum(img, axis=2) > 0
    # Mask3[:, :, 1] = np.sum(img, axis=2) > 0  # original: img[..., 0] > 0
    # Mask3[:, :, 2] = np.sum(img, axis=2) > 0  # original: img[..., 0] > 0

    # get dye image
    L_Obj = np.zeros((3, Img_size))

    L_Obj[0, :] = L_Mel
    L_Obj[1, :] = L_Hem
    L_Obj[2, :] = L_Shd

    L_Vec = np.hstack([L_Obj[0, :], L_Obj[1, :], L_Obj[2, :]])

    img2 = np.reshape(L_Vec, (3, width, height)).T
    img_e = np.exp(-img2)
    img_exp = np.clip(img_e, 0, 1)[:, :, ::-1]
    # ef_img = img_exp * Mask3.astype(np.float32)
    ef_img = img_exp.astype(np.float32)

    return ef_img


# }}}


def pigment_imgs(img, melanin=JAP_MEL_VEC, hemoglobin=JAP_HEM_VEC):
    # {{{
    """
    [10] Transformation of color components into image space
    Returns a 3 images with 3 channels each, representing the pigments
    """

    L_Mel, L_Hem, L_Shd = pigment_separation(img, melanin, hemoglobin)

    height, width, channels = img.shape[:3]
    Img_size = height * width

    # Mask3 = np.empty((height, width, 3), dtype=np.bool)
    # Mask3[:, :, 0] = np.sum(img, axis=2) > 0
    # Mask3[:, :, 1] = np.sum(img, axis=2) > 0  # original: img[..., 0] > 0
    # Mask3[:, :, 2] = np.sum(img, axis=2) > 0  # original: img[..., 0] > 0

    # get dye image
    L_ObjHem = np.zeros((3, Img_size))
    L_ObjMel = np.zeros((3, Img_size))
    L_ObjShd = np.zeros((3, Img_size))

    for j in range(3):
        L_ObjHem[j, :] = hemoglobin[j] * L_Hem

    for j in range(3):
        L_ObjMel[j, :] = melanin[j] * L_Mel

    for j in range(3):
        L_ObjShd[j, :] = L_Shd

    L_VecHem = np.hstack([L_ObjHem[0, :], L_ObjHem[1, :], L_ObjHem[2, :]])

    img2 = np.reshape(L_VecHem, (3, width, height)).T
    img_e = np.exp(-img2)
    img_exp = np.clip(img_e, 0, 1)[:, :, ::-1]
    ef_img_hem = img_exp.astype(np.float32)

    L_VecMel = np.hstack([L_ObjMel[0, :], L_ObjMel[1, :], L_ObjMel[2, :]])

    img2 = np.reshape(L_VecMel, (3, width, height)).T
    img_e = np.exp(-img2)
    img_exp = np.clip(img_e, 0, 1)[:, :, ::-1]
    ef_img_mel = img_exp.astype(np.float32)

    L_VecShd = np.hstack([L_ObjShd[0, :], L_ObjShd[1, :], L_ObjShd[2, :]])

    img2 = np.reshape(L_VecShd, (3, width, height)).T
    img_e = np.exp(-img2)
    img_exp = np.clip(img_e, 0, 1)[:, :, ::-1]
    ef_img_shd = img_exp.astype(np.float32)

    return (ef_img_mel, ef_img_hem, ef_img_shd)


# }}}
