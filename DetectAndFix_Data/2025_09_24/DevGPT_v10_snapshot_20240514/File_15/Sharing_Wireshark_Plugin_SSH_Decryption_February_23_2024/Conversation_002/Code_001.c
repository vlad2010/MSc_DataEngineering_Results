#include <epan/packet.h>
#include <epan/proto.h>
#include <epan/dissectors/packet-ssh.h>
#include "ssh_utils.h"
#include "ssh_key_handler.h"

#define SSH_DECRYPTOR_VERSION "1.0"
#define SSH_DECRYPTOR_DESCRIPTION "SSH Traffic Decryptor"

static int proto_ssh_decryptor = -1;
static int hf_ssh_decrypted_payload = -1;

static gint ett_ssh_decryptor = -1;

static void dissect_ssh_decryptor(tvbuff_t *tvb, packet_info *pinfo, proto_tree *tree, void *data)
{
    col_set_str(pinfo->cinfo, COL_PROTOCOL, "SSHDECRYPTOR");
    col_clear(pinfo->cinfo,COL_INFO);

    proto_item *ti = proto_tree_add_item(tree, proto_ssh_decryptor, tvb, 0, -1, ENC_NA);
    proto_tree *ssh_decryptor_tree = proto_item_add_subtree(ti, ett_ssh_decryptor);

    // Assuming the key has already been loaded and is available
    guchar *decrypted_data;
    guint decrypted_data_len = 0;
    if (decrypt_ssh_payload(tvb_get_ptr(tvb, 0, -1), tvb_reported_length(tvb), &decrypted_data, &decrypted_data_len)) {
        proto_tree_add_bytes(ssh_decryptor_tree, hf_ssh_decrypted_payload, tvb, 0, decrypted_data_len, decrypted_data);
        col_add_fstr(pinfo->cinfo, COL_INFO, "SSH Decrypted Payload, Len: %u", decrypted_data_len);
    } else {
        col_add_fstr(pinfo->cinfo, COL_INFO, "SSH Decryption Failed");
    }
}

void proto_register_ssh_decryptor(void)
{
    static hf_register_info hf[] = {
        { &hf_ssh_decrypted_payload,
          { "Decrypted Payload", "sshdecryptor.decrypted_payload",
            FT_BYTES, BASE_NONE, NULL, 0x0,
            NULL, HFILL }
        },
    };

    static gint *ett[] = {
        &ett_ssh_decryptor,
    };

    proto_ssh_decryptor = proto_register_protocol(
        SSH_DECRYPTOR_DESCRIPTION,
        "SSHDECRYPTOR",
        "sshdecryptor"
    );

    proto_register_field_array(proto_ssh_decryptor, hf, array_length(hf));
    proto_register_subtree_array(ett, array_length(ett));
}

void proto_reg_handoff_ssh_decryptor(void)
{
    static dissector_handle_t ssh_decryptor_handle;

    ssh_decryptor_handle = create_dissector_handle(dissect_ssh_decryptor, proto_ssh_decryptor);
    dissector_add_uint("ssh.port", 22, ssh_decryptor_handle);
}
