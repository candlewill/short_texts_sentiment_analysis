import qrcode
import scipy


def to_qrcode(texts):
    qr = qrcode.QRCode(
        version=7,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=0,
    )
    N = 140
    if len(texts) <= N:
        s = texts
    else:
        s = texts[:N]
    qr.add_data(s)
    qr.make(fit=False)
    img = qr.make_image()
    return img


if __name__ == '__main__':
    img = to_qrcode('HelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHellolloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHello')
    a = list(img.getdata())
    print(len(a), a)
    with open('./data/pic/test.png', 'wb') as f:
        img.save(f)