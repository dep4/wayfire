#include "wayfire/unstable/wlr-surface-node.hpp"
#include "pixman.h"
#include "view/view-impl.hpp"
#include "wayfire/geometry.hpp"
#include "wayfire/opengl.hpp"
#include "wayfire/region.hpp"
#include "wayfire/render-manager.hpp"
#include "wayfire/scene-render.hpp"
#include "wayfire/scene.hpp"
#include "wlr-surface-pointer-interaction.hpp"
#include "wlr-surface-touch-interaction.cpp"
#include <GLES3/gl3.h>
#include <memory>
#include <sstream>
#include <string>
#include <wayfire/signal-provider.hpp>
#include <wlr/util/box.h>

extern bool disable_gl_call;

wf::scene::surface_state_t::surface_state_t(surface_state_t&& other)
{
    if (&other != this)
    {
        *this = std::move(other);
    }
}

wf::scene::surface_state_t& wf::scene::surface_state_t::operator =(surface_state_t&& other)
{
    if (current_buffer)
    {
        wlr_buffer_unlock(current_buffer);
    }

    current_buffer = other.current_buffer;
    texture = other.texture;
    accumulated_damage = other.accumulated_damage;
    size = other.size;
    src_viewport = other.src_viewport;

    other.current_buffer = NULL;
    other.texture = NULL;
    other.accumulated_damage.clear();
    other.src_viewport.reset();
    return *this;
}

void wf::scene::surface_state_t::merge_state(wlr_surface *surface)
{
    // NB: lock the new buffer first, in case it is the same as the old one
    if (surface->buffer)
    {
        wlr_buffer_lock(&surface->buffer->base);
    }

    if (current_buffer)
    {
        wlr_buffer_unlock(current_buffer);
    }

    if (surface->buffer)
    {
        this->current_buffer = &surface->buffer->base;
        this->texture = surface->buffer->texture;
        this->size    = {surface->current.width, surface->current.height};
    } else
    {
        this->current_buffer = NULL;
        this->texture = NULL;
        this->size    = {0, 0};
    }

    if (surface->current.viewport.has_src)
    {
        wlr_fbox fbox;
        wlr_surface_get_buffer_source_box(surface, &fbox);
        this->src_viewport = fbox;
    } else
    {
        this->src_viewport.reset();
    }

    wf::region_t current_damage;
    wlr_surface_get_effective_damage(surface, current_damage.to_pixman());
    this->accumulated_damage |= current_damage;
}

wf::scene::surface_state_t::~surface_state_t()
{
    if (current_buffer)
    {
        wlr_buffer_unlock(current_buffer);
    }
}

wf::scene::wlr_surface_node_t::wlr_surface_node_t(wlr_surface *surface, bool autocommit) :
    node_t(false), autocommit(autocommit)
{
    this->surface = surface;
    this->ptr_interaction = std::make_unique<wlr_surface_pointer_interaction_t>(surface, this);
    this->tch_interaction = std::make_unique<wlr_surface_touch_interaction_t>(surface);

    this->on_surface_destroyed.set_callback([=] (void*)
    {
        this->surface = NULL;
        this->ptr_interaction = std::make_unique<pointer_interaction_t>();
        this->tch_interaction = std::make_unique<touch_interaction_t>();

        on_surface_commit.disconnect();
        on_surface_destroyed.disconnect();
    });

    this->on_surface_commit.set_callback([=] (void*)
    {
        if (!wlr_surface_has_buffer(this->surface) && this->visibility.empty())
        {
            send_frame_done(false);
        }

        if (this->autocommit)
        {
            apply_current_surface_state();
        }

        for (auto& [wo, _] : visibility)
        {
            wo->render->schedule_redraw();
        }
    });

    on_surface_destroyed.connect(&surface->events.destroy);
    on_surface_commit.connect(&surface->events.commit);
    send_frame_done(false);

    current_state.merge_state(surface);
}

void wf::scene::wlr_surface_node_t::apply_state(surface_state_t&& state)
{
    const bool size_changed = current_state.size != state.size;
    this->current_state = std::move(state);
    wf::scene::damage_node(this, current_state.accumulated_damage);
    if (size_changed)
    {
        scene::update(this->shared_from_this(), scene::update_flag::GEOMETRY);
    }
}

void wf::scene::wlr_surface_node_t::apply_current_surface_state()
{
    surface_state_t state;
    state.merge_state(surface);
    this->apply_state(std::move(state));
}

std::optional<wf::scene::input_node_t> wf::scene::wlr_surface_node_t::find_node_at(const wf::pointf_t& at)
{
    if (!surface)
    {
        return {};
    }

    if (wlr_surface_point_accepts_input(surface, at.x, at.y))
    {
        wf::scene::input_node_t result;
        result.node = this;
        result.local_coords = at;
        return result;
    }

    return {};
}

std::string wf::scene::wlr_surface_node_t::stringify() const
{
    std::ostringstream name;
    name << "wlr-surface-node ";
    if (surface)
    {
        name << "surface";
    } else
    {
        name << "inert";
    }

    name << " " << stringify_flags();
    return name.str();
}

wf::pointer_interaction_t& wf::scene::wlr_surface_node_t::pointer_interaction()
{
    return *this->ptr_interaction;
}

wf::touch_interaction_t& wf::scene::wlr_surface_node_t::touch_interaction()
{
    return *this->tch_interaction;
}

void wf::scene::wlr_surface_node_t::send_frame_done(bool delay_until_vblank)
{
    if (!surface)
    {
        return;
    }

    if (!delay_until_vblank || visibility.empty())
    {
        timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        wlr_surface_send_frame_done(surface, &now);
    } else
    {
        for (auto& [wo, _] : visibility)
        {
            wlr_output_schedule_frame(wo->handle);
        }
    }
}

static const char *batch_vs =
    R"(#version 320 es

in mediump vec2 base_model;

in mediump vec4 position;
in mediump vec4 in_uvpos;

in uint in_texid;

out mediump vec2 uvpos;
flat out uint texid;

void main() {
    gl_Position = vec4(base_model * position.xy + position.zw, 0.0, 1.0);
    uvpos = base_model * in_uvpos.xy + in_uvpos.zw;
    texid = in_texid;
})";

static const char *batch_fs =
    R"(#version 320 es

uniform sampler2D textures[8];
in mediump vec2 uvpos;
flat in uint texid;

layout(location = 0) out mediump vec4 fragcolor;

void main() {
    if (texid == 0u) {
        fragcolor = vec4(1.0, 0.0, 0.0, 1.0) * vec4(uvpos, 0.0, 1.0);
    } else {
        fragcolor = vec4(1.0, 1.0, 0.0, 1.0) * vec4(uvpos, 0.0, 1.0);
    }

    //fragcolor = vec4(uvpos, 0.0, 1.0);

    fragcolor = texture2D(textures[texid], uvpos);
})";

OpenGL::program_t batch_prog;

class wf::scene::wlr_surface_node_t::wlr_surface_render_instance_t : public render_instance_t
{
    std::shared_ptr<wlr_surface_node_t> self;
    wf::signal::connection_t<wf::frame_done_signal> on_frame_done = [=] (wf::frame_done_signal *ev)
    {
        self->send_frame_done(false);
    };

    wf::output_t *visible_on;
    damage_callback push_damage;

    wf::signal::connection_t<node_damage_signal> on_surface_damage =
        [=] (node_damage_signal *data)
    {
        if (self->surface)
        {
            // Make sure to expand damage, because stretching the surface may cause additional damage.
            const float scale = self->surface->current.scale;
            const float output_scale = visible_on ? visible_on->handle->scale : 1.0;
            if (scale != output_scale)
            {
                data->region.expand_edges(std::ceil(std::abs(scale - output_scale)));
            }
        }

        push_damage(data->region);
    };

  public:
    wlr_surface_render_instance_t(std::shared_ptr<wlr_surface_node_t> self,
        damage_callback push_damage, wf::output_t *visible_on)
    {
        if (visible_on)
        {
            self->visibility[visible_on]++;
            if (self->surface)
            {
                wlr_surface_send_enter(self->surface, visible_on->handle);
            }
        }

        this->self = self;
        this->push_damage = push_damage;
        this->visible_on  = visible_on;
        self->connect(&on_surface_damage);
    }

    ~wlr_surface_render_instance_t()
    {
        if (visible_on)
        {
            self->visibility[visible_on]--;
            if ((self->visibility[visible_on] == 0) && self->surface)
            {
                self->visibility.erase(visible_on);
                wlr_surface_send_leave(self->surface, visible_on->handle);
            }
        }
    }

    void schedule_instructions(std::vector<render_instruction_t>& instructions,
        const wf::render_target_t& target, wf::region_t& damage) override
    {
        wf::region_t our_damage = damage & self->get_bounding_box();
        if (!our_damage.empty())
        {
            instructions.push_back(render_instruction_t{
                .instance = this,
                .target   = target,
                .damage   = std::move(our_damage),
            });

            if (self->surface)
            {
                pixman_region32_subtract(damage.to_pixman(), damage.to_pixman(),
                    &self->surface->opaque_region);
            }
        }
    }

    void render(const wf::render_target_t& target, const wf::region_t& region) override
    {}

    void presentation_feedback(wf::output_t *output) override
    {
        if (self->surface)
        {
            wlr_presentation_surface_sampled_on_output(wf::get_core_impl().protocols.presentation,
                self->surface, output->handle);
        }
    }

    direct_scanout try_scanout(wf::output_t *output) override
    {
        if (!self->surface)
        {
            return direct_scanout::SKIP;
        }

        if (self->get_bounding_box() != output->get_relative_geometry())
        {
            return direct_scanout::OCCLUSION;
        }

        // Must have a wlr surface with the correct scale and transform
        auto wlr_surf = self->surface;
        if ((wlr_surf->current.scale != output->handle->scale) ||
            (wlr_surf->current.transform != output->handle->transform))
        {
            return direct_scanout::OCCLUSION;
        }

        // Finally, the opaque region must be the full surface.
        wf::region_t non_opaque = output->get_relative_geometry();
        non_opaque ^= wf::region_t{&wlr_surf->opaque_region};
        if (!non_opaque.empty())
        {
            return direct_scanout::OCCLUSION;
        }

        wlr_presentation_surface_sampled_on_output(
            wf::get_core().protocols.presentation, wlr_surf, output->handle);
        wlr_output_attach_buffer(output->handle, &wlr_surf->buffer->base);

        if (wlr_output_commit(output->handle))
        {
            return direct_scanout::SUCCESS;
        } else
        {
            return direct_scanout::OCCLUSION;
        }
    }

    void compute_visibility(wf::output_t *output, wf::region_t& visible) override
    {
        auto our_box = self->get_bounding_box();
        on_frame_done.disconnect();

        if (!(visible & our_box).empty())
        {
            // We are visible on the given output => send wl_surface.frame on output frame, so that clients
            // can draw the next frame.
            output->connect(&on_frame_done);
            // TODO: compute actually visible region and disable damage reporting for that region.
        }
    }

    class wlr_surface_batch : public wf::scene::render_batch_t
    {
      public:
        GLuint fb;

        wlr_surface_batch(GLuint fb)
        {
            this->fb = fb;

            static int count = 0;
            if (count == 0)
            {
                batch_prog.set_simple(OpenGL::compile_program(batch_vs, batch_fs));
                count = 1;
            }

            // std::_Exit(-1);
        }

        void submit()
        {
            // We don't expect any errors from us!
            // disable_gl_call = true;

            int changes = 0;
            std::map<GLuint, GLuint> bound_textures;
            std::vector<GLfloat> vertexData;
            std::vector<GLfloat> coordData;
            std::vector<GLuint> texData;

            for (auto& i : instructions)
            {
                auto self = dynamic_cast<wlr_surface_render_instance_t*>(i.instance)->self;
                if (!self->current_state.current_buffer)
                {
                    continue;
                }

                wf::geometry_t geometry = self->get_bounding_box();
                i.damage &= geometry;

                wf::texture_t tex{self->current_state.texture, self->current_state.src_viewport};

                GLuint tex_id;
                if (bound_textures.count(tex.tex_id))
                {
                    tex_id = bound_textures[tex.tex_id];
                } else
                {
                    auto size = bound_textures.size();
                    bound_textures[tex.tex_id] = size;
                    tex_id = size;
                }

                glm::mat4 matrix = i.target.get_orthographic_projection();
                for (const auto& rect : i.damage)
                {
                    glm::vec4 xy   = matrix * glm::vec4(1.0 * rect.x1, 1.0 * rect.y1, 0.0, 1.0);
                    glm::vec4 xywh = matrix * glm::vec4(1.0 * rect.x2, 1.0 * rect.y2, 0.0, 1.0);

                    // Calculate a triangle strip:
                    // 4 - 3
                    // |   |
                    // 1 - 2

                    vertexData.insert(vertexData.end(), {
                        xywh.x - xy.x, xy.y - xywh.y,
                        xy.x, xywh.y,
                    });

                    // vertexData.insert(vertexData.end(), {
                    // xy.x, xywh.y,
                    // xywh.x, xywh.y,
                    // xywh.x, xy.y,
                    // xy.x, xy.y,
                    // });

                    // from bottom left to top right
                    gl_geometry tex_g = {
                        .x1 = 1.0f * (rect.x1 - geometry.x) / geometry.width,
                        .y1 = 1.0f * (geometry.y + geometry.height - rect.y2) / geometry.height,
                        .x2 = 1.0f * (rect.x2 - geometry.x) / geometry.width,
                        .y2 = 1.0f * (geometry.y + geometry.height - rect.y1) / geometry.height,
                    };

                    if (tex.invert_y)
                    {
                        tex_g.y1 = 1 - tex_g.y1;
                        tex_g.y2 = 1 - tex_g.y2;
                    }

                    // LOGI("Rendering ", geometry, " ", wlr_box_from_pixman_box(rect), " ",
                    // tex_g.x1, " ", tex_g.y1, " ", tex_g.x2, " ", tex_g.y2);

                    // LOGI(std::abs(tex_g.x1 - tex_g.x2), " ", std::abs(tex_g.y1 - tex_g.y2), " ",
                    // (tex_g.x1 + tex_g.x2) / 2.0f, " ", (tex_g.y1 + tex_g.y2) / 2.0f);

                    coordData.insert(coordData.end(), {
                        tex_g.x2 - tex_g.x1, tex_g.y2 - tex_g.y1,
                        tex_g.x1, tex_g.y1,
                    });

                    // coordData.insert(coordData.end(), {
                    // tex_g.x1, tex_g.y1,
                    // tex_g.x2, tex_g.y1,
                    // tex_g.x2, tex_g.y2,
                    // tex_g.x1, tex_g.y2,
                    // });

                    texData.push_back(tex_id);
                    // texData.push_back(tex_id);
                    // texData.push_back(tex_id);
                    // texData.push_back(tex_id);
                }
            }

            OpenGL::render_begin(instructions[0].target);
            GL_CALL(glEnable(GL_BLEND));
            GL_CALL(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));

            batch_prog.use(TEXTURE_TYPE_RGBA);
            for (auto [tex, id] : bound_textures)
            {
                GL_CALL(glActiveTexture(GL_TEXTURE0 + id));
                GL_CALL(glBindTexture(GL_TEXTURE_2D, tex));
                GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            }

            for (int i = 0; i < 8; i++)
            {
                std::string name = "textures[" + std::to_string(i) + "]";
                batch_prog.uniform1i(name, i);
            }

            static const GLfloat base_model[] = {
                0, 0,
                1, 0,
                1, 1,
                0, 1,
            };

            batch_prog.attrib_pointer("base_model", 2, 0, base_model);

            batch_prog.attrib_pointer("position", 4, 0, vertexData.data());
            batch_prog.attrib_pointer("in_uvpos", 4, 0, coordData.data());
            batch_prog.attrib_ipointer("in_texid", 1, 0, texData.data(), GL_UNSIGNED_INT);

            batch_prog.attrib_divisor("position", 1);
            batch_prog.attrib_divisor("in_uvpos", 1);
            batch_prog.attrib_divisor("in_texid", 1);

            GL_CALL(glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, texData.size()));

            // Unset textures
            for (auto [tex, id] : bound_textures)
            {
                GL_CALL(glActiveTexture(GL_TEXTURE0 + id));
                GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));
            }

            GL_CALL(glActiveTexture(GL_TEXTURE0));

            OpenGL::render_end();
            LOGI("Batch changed ", changes);
        }

        void add(wlr_surface_render_instance_t *rinst, scene::render_instruction_t& instr)
        {
            instructions.push_back(instr);
        }

        std::vector<scene::render_instruction_t> instructions;
    };

    bool can_batch() override
    {
        return true;
    }

    std::unique_ptr<render_batch_t> start_batch(scene::render_instruction_t& instr) override
    {
        auto batch = std::make_unique<wlr_surface_batch>(instr.target.fb);
        batch->add(this, instr);
        return batch;
    }

    bool try_add_to_batch(render_batch_t *batch, scene::render_instruction_t& instr) override
    {
        if (auto vbatch = dynamic_cast<wlr_surface_batch*>(batch))
        {
            if (vbatch->fb != instr.target.fb)
            {
                return false;
            }

            wf::texture_t tex{self->current_state.texture, self->current_state.src_viewport};
            if (tex.type != TEXTURE_TYPE_RGBA)
            {
                return false;
            }

            vbatch->add(this, instr);
            return true;
        }

        return false;
    }
};

void wf::scene::wlr_surface_node_t::gen_render_instances(
    std::vector<render_instance_uptr>& instances, damage_callback damage,
    wf::output_t *output)
{
    instances.push_back(std::make_unique<wlr_surface_render_instance_t>(
        std::dynamic_pointer_cast<wlr_surface_node_t>(this->shared_from_this()), damage, output));
}

wf::geometry_t wf::scene::wlr_surface_node_t::get_bounding_box()
{
    return wf::construct_box({0, 0}, current_state.size);
}

wlr_surface*wf::scene::wlr_surface_node_t::get_surface() const
{
    return this->surface;
}

std::optional<wf::texture_t> wf::scene::wlr_surface_node_t::to_texture() const
{
    if (this->current_state.current_buffer)
    {
        return wf::texture_t{current_state.texture, current_state.src_viewport};
    }

    return {};
}
